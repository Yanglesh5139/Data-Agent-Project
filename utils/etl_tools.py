import os
import requests
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Callable
from urllib.parse import urlparse




class ETLTools:
    """Dynamic ETL utility for extracting data from URLs and persisting it."""

    # Supported output formats and their writer functions
    WRITERS: Dict[str, Callable[[pd.DataFrame, str], None]] = {
        "csv": lambda df, path: df.to_csv(path, index=False),
        "json": lambda df, path: df.to_json(path, orient="records", lines=True),
        "parquet": lambda df, path: df.to_parquet(path, index=False),
        "xlsx": lambda df, path: df.to_excel(path, index=False),
    }
    READERS = {
    "csv":     pd.read_csv,
    "json":    lambda p: pd.read_json(p, orient="records", lines=True),
    "parquet": pd.read_parquet,
    "xlsx":    pd.read_excel,
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ):
        """
        Args:
            base_dir: Root directory for relative output paths.
                      Defaults to the project root (parent of this file's dir).
            timeout: HTTP request timeout in seconds.
            session: Optional requests.Session for connection reuse.
        """
        self.base_dir = base_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def extract_python_code(response) -> str:
        content = response.content

        if isinstance(content, str):
            code = content
        elif isinstance(content, list):
            code = "\n".join(
                block if isinstance(block, str) else block["text"]
                for block in content
                if isinstance(block, str)
                or (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                )
            )
        else:
            raise TypeError(
                f"Unexpected content type: {type(content).__name__}"
            )

        code = code.strip()
        lines = code.splitlines()

        if (
            len(lines) >= 2
            and lines[0].strip() in {"```python", "```"}
            and lines[-1].strip() == "```"
        ):
            code = "\n".join(lines[1:-1]).strip()

        if not code:
            raise ValueError("The model returned no Python code.")

        return code
    def _upsert(self, df_new: pd.DataFrame, path: str, fmt: str,
        key_cols: Union[str, List[str]]) -> pd.DataFrame:
        """Merge new data into existing file, updating on key_cols."""
        keys = [key_cols] if isinstance(key_cols, str) else key_cols
        for k in keys:
            if k not in df_new.columns:
                raise KeyError(f"Key column '{k}' not in new data")

        if not os.path.exists(path):
            self.WRITERS[fmt](df_new, path)
            return df_new

        df_old = self.READERS[fmt](path)

        # Drop old rows whose keys appear in the new data, then concat
        if all(k in df_old.columns for k in keys):
            merged = pd.concat(
                [
                    df_old.merge(df_new[keys].drop_duplicates(),
                                on=keys, how="left", indicator=True)
                        .query("_merge == 'left_only'")
                        .drop(columns="_merge"),
                    df_new,
                ],
                ignore_index=True,
            )
        else:
            # Schema changed (new key columns) — fall back to append
            merged = pd.concat([df_old, df_new], ignore_index=True)

        self.WRITERS[fmt](merged, path)
        return merged

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_load(
        self,
        url: Union[str, List[str]],
        output_path: str,
        format: str = "csv",
        mode: str = "overwrite", # "overwrite" | "append" | "upsert"
        json_key: Optional[Union[str, List[str]]] = None,
        key_cols: Optional[Union[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None,
        filename: Optional[str] = None,
        **request_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract JSON data from one or more URLs and save it to disk.

        Args:
            url: A single URL string or list of URL strings.
            output_path: Destination directory (relative to base_dir).
            format: Output format - one of 'csv', 'json', 'parquet', 'xlsx'.
            json_key: Optional key (or path of keys) to drill into the
                      JSON payload before normalizing. If None, the whole
                      payload is used.
            headers: Optional HTTP headers.
            **request_kwargs: Additional kwargs passed to requests.get().

        Returns:
            A dict summarizing the results per URL.
        """
        if format not in self.WRITERS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Choose from: {', '.join(self.WRITERS)}"
            )

        urls = [url] if isinstance(url, str) else list(url)
        if not urls:
            raise ValueError("No URLs provided.")

        output_dir = self._resolve_output_dir(output_path)
        os.makedirs(output_dir, exist_ok=True)

        results: Dict[str, Any] = {}

        for idx, current_url in enumerate(urls):
            try:
                df = self._fetch_and_normalize(
                    current_url, json_key=json_key,
                    headers=headers, **request_kwargs
                )
                file_name = self._build_file_name(
                    current_url, output_dir, format, idx, len(urls),explicit=filename
                )
                if mode == "overwrite":
                    self.WRITERS[format](df, file_name)
                elif mode == "append":
                    self._append(df, file_name, format)
                elif mode == "upsert":
                    if key_cols is None:
                        raise ValueError("mode='upsert' requires key_cols")
                    df = self._upsert(df, file_name, format, key_cols)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                results[current_url] = {"status": "success",
                                        "file": file_name,
                                        "rows": len(df)}
            except Exception as exc:
                print(f"Error occurred while processing {current_url}: {exc}")
                results[current_url] = {"status": "error", "error": str(exc)}

        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _fetch_and_normalize(
        self,
        url: str,
        json_key: Optional[Union[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None,
        **request_kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch JSON from a URL and normalize it into a DataFrame."""
        response = self.session.get(
            url, headers=headers, timeout=self.timeout, **request_kwargs
        )
        response.raise_for_status()
        data = response.json()

        if json_key is not None:
            data = self._drill_down(data, json_key)

        # Normalize whatever structure we got
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # If top-level dict contains a single list value, use it
            list_values = [v for v in data.values() if isinstance(v, list)]
            if len(list_values) == 1:
                df = pd.json_normalize(list_values[0])
            else:
                df = pd.json_normalize(data)
        else:
            raise ValueError(f"Unexpected JSON root type: {type(data)}")

        return df

    @staticmethod
    def _drill_down(data: Any, keys: Union[str, List[str]]) -> Any:
        """Walk into nested dict keys (e.g. ['data', 'breeds'])."""
        path = [keys] if isinstance(keys, str) else keys
        for k in path:
            if not isinstance(data, dict) or k not in data:
                raise KeyError(f"Key '{k}' not found in JSON payload")
            data = data[k]
        return data

    def _resolve_output_dir(self, output_path: str) -> str:
        """Resolve relative output paths against base_dir."""
        return (
            output_path
            if os.path.isabs(output_path)
            else os.path.join(self.base_dir, output_path)
        )

    @staticmethod
    def _build_file_name(
        url: str, output_dir: str, fmt: str, idx: int, total: int, explicit: Optional[str] = None
    ) -> str:
        """
        Build a unique file name per URL.
        Single URL -> extracted_data.<fmt>
        Multiple    -> <url_basename>_<idx>.<fmt>
        """
        if explicit:
            return os.path.join(output_dir, f"{explicit}.{fmt}")
        try:
            parsed = urlparse(url)
            stem = os.path.splitext(os.path.basename(parsed.path))[0] or "data"
            stem = f"{stem}_{idx}"
        except Exception:
            print(f"Warning: Could not parse URL '{url}' for filename. Using default.")
            stem = f"extracted_data_{idx}"
        return os.path.join(output_dir, f"{stem}.{fmt}")

    def transform_load_context(self,input_file_path:str,output_path:str,format:str):
        """
        Transform and load data from a file to a specified output format.

        Args:
            input_file_path: Path to the input file.
            output_path: Destination directory (relative to base_dir).
            format: Output format - one of 'csv', 'json', 'parquet', 'xlsx'.
        """
        if format not in self.WRITERS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Choose from: {', '.join(self.WRITERS)}"
            )

        output_dir = self._resolve_output_dir(output_path)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Read the input file based on its extension
            ext = os.path.splitext(input_file_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(input_file_path)
            elif ext == ".json":
                df = pd.read_json(input_file_path, orient="records", lines=True)
            elif ext == ".parquet":
                df = pd.read_parquet(input_file_path)
            elif ext == ".xlsx":
                df = pd.read_excel(input_file_path)
            else:
                raise ValueError(f"Unsupported input file extension: {ext}")

            top_5_rows=str(df.head(5))

            return top_5_rows
        except Exception as exc:
            print(f"Error occurred while transforming and loading data: {exc}")

    def execute_code(self, code: str) -> str:
        """Execute generated Python. This is NOT a security sandbox."""
        if not isinstance(code, str):
            raise TypeError(
                f"Expected Python source text, got {type(code).__name__}"
            )

        # Arbitrary Python execution: use only in a trusted local prototype.
        # Use an isolated execution worker for untrusted inputs.
        namespace = {}
        exec(compile(code, "<etl-generated>", "exec"), namespace, namespace)

        return "Code executed successfully."

if __name__ == "__main__":
    etl = ETLTools()
    # Example 1: single URL, drill into the "breeds" key
    result = etl.extract_load(
        url=["https://breedhealthscore.com/api/breeds.json"],
        output_path="data/extracted",
        format="csv",
        mode="overwrite"
    )
    print(etl.transform_load_context("data/extracted/breeds_0.csv"))


   

    