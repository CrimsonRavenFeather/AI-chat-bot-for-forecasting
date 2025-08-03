from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.LLMS.groq import GROQ_LLM
from src.langgraphagenticai.tools.machine_learning_tools import MachineLearningTool
from langchain_core.messages import AIMessage,HumanMessage,FunctionMessage
import pandas as pd

dataset = pd.read_csv("src\langgraphagenticai\DATASETS\Screentime - App Details.csv")
machine_learning_tool=MachineLearningTool(dataset)
function_map={}

function_map["dummy_machine_learning_tools"]=machine_learning_tool.process

class ChatNode:
    def __init__(self,llm):
        self.llm=llm

    def chat_node(self, state: State):
        groq_llm_model = self.llm
        result = groq_llm_model.invoke(state["messages"])

        if hasattr(result, "tool_calls") and result.tool_calls:
            # Store original LLM message that triggered tool call
            state["messages"].append(result)

            function_name = result.tool_calls[0]["name"]
            callable_function = function_map.get(function_name)
            
            if callable_function:
                response = callable_function()
                function_message = FunctionMessage(content=str(response), name=function_name)
                state["messages"].append(function_message)

                # OPTIONAL: You can now re-invoke LLM with updated state
                follow_up = groq_llm_model.invoke(state["messages"])
                state["messages"].append(follow_up)  # AI responds to tool output
        else:
            state["messages"].append(result)
        
        return state  # Return updated state

