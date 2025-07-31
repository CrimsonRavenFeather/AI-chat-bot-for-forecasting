import os
from langchain_groq import ChatGroq
from src.langgraphagenticai.state.state import State
from langchain_core.messages import AIMessage,HumanMessage,FunctionMessage
from langchain_core.tools import tool

# DUMMY TOOLS
@tool
def dummy_machine_learning_tools(dataset:str)->dict:
    """_summary_
    classify the dataset and run basic machine learning models
    Args:
        dataset (str): Name of the dataset

    Returns:
        dict: type of dataset and  stats of basic machine learning models for the give dataset
    """
    return {} 

tools=[dummy_machine_learning_tools]

class GROQ_LLM:
    def __init__(self,model,api):
        self.model=model
        self.api=api

    def get_llm_model(self):
        try:
            llm = ChatGroq(api_key=self.api,model=self.model)
            llm_with_tools = llm.bind_tools(tools)
        except Exception as e:
            raise ValueError(f"Error occured with exception : {e}")
        return llm_with_tools
        
