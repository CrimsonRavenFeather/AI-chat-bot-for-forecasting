from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.LLMS.groq import GROQ_LLM
from src.langgraphagenticai.tools.machine_learning_tools import MachineLearningTool
from langchain_core.messages import AIMessage,HumanMessage,FunctionMessage
import pandas as pd

dataset = pd.read_csv("src\langgraphagenticai\DATASETS\Screentime - App Details.csv")
machine_learning_tool=MachineLearningTool(dataset)
function_map={}

function_map["dummy_machine_learning_tools"]=machine_learning_tool.process

def chat_node(model,api,state:State):
    print("RUNNING LLM MODEL")
    groq_llm = GROQ_LLM(model,api)
    groq_llm_model=groq_llm.get_llm_model()
    result=groq_llm_model.invoke(state["messages"])
    if result.tool_calls:
        function_name=result.tool_calls[0]["name"]
        callable_function=function_map[function_name]
        response=callable_function()
        function_message=FunctionMessage(content=response,name=function_name)
        state["messages"].append(function_message)
    else:
        state["messages"].append(result)
