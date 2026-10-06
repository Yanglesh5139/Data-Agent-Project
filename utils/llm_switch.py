from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

def switch_llm(level :str):
    """
    Switches the LLM based on the provided level.

    Args:
        level (str): The level of the LLM to switch to. 
                     Options are 'low', 'medium', 'high'.

    Returns:
        ChatOpenAI: The initialized ChatOpenAI instance for the specified level.
    """
    if level.lower() == 'low':
       llm=ChatOpenAI(model_name="gpt-5.6-luna",temperature=0,max_tokens=2000)
    elif level.lower() == 'medium':
        llm=ChatOpenAI(model_name="gpt-5.6-terra",temperature=0,max_tokens=2000)
    elif level.lower() == 'high':
        llm=ChatOpenAI(model_name="gpt-5.6-sol",temperature=0,max_tokens=2000)
    else:
        raise ValueError(f"Unsupported level: {level}")

    return llm

if __name__=="__main__":

    llm_obj=switch_llm("low")
    print(llm_obj.invoke("What is the capital of Italy?"))