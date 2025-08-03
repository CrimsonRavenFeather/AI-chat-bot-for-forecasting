from langgraph.graph import StateGraph,START,END
from src.langgraphagenticai.LLMS.groq import GROQ_LLM
from src.langgraphagenticai.nodes.chat_node import ChatNode
from src.langgraphagenticai.state.state import State

class GraphBuilder:
    def __init__(self,model,api):
        llm_wrapper = GROQ_LLM(model=model, api=api)
        llm_runnable = llm_wrapper.get_llm_model()  # This returns ChatGroq().bind_tools(...)

        self.llm = llm_runnable
        self.graph_builder = StateGraph(State)
    
    def chatbot_graph_builder(self):
        self.chatbot_node = ChatNode(llm=self.llm)

        self.graph_builder.add_node("chatbot", self.chatbot_node.chat_node)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)

        return self.graph_builder.compile()
