import os
from langchain_groq import ChatGroq
from src.langgraphagenticai.state.state import State
from langchain_core.messages import AIMessage,HumanMessage,FunctionMessage

class GROQ_LLM:
    def __init__(self,model):
        self.model=model
        self.llm_groq=ChatGroq(model=model)

    def chat_with_groq(self,state:State):
        return self.llm_groq(state['messages'])
