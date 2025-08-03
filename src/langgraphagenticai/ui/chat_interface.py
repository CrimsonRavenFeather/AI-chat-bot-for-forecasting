import streamlit as st
from src.langgraphagenticai.state.state import State
from langchain_core.messages import AIMessage,HumanMessage
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
import os
from dotenv import load_dotenv

load_dotenv()

class StreamlitUI:
    def __init__(self):
        self.state=State()
        self.state["messages"]=[AIMessage(content="👋 Hello! I'm your Time Series Forecasting Assistant.  How may I help you?",name="ai"),HumanMessage(content="How many stars are there in out galaxy")]
        self.initial_greetings=False

    def hello_display(self):
        st.title("Hello Streamlit-er 👋")
        st.markdown(
            """ 
            This is a playground for you to try Streamlit and have fun. 

            **There's :rainbow[so much] you can build!**
            
            We prepared a few examples for you to get started. Just 
            click on the buttons above and discover what you can do 
            with Streamlit. 
            """
        )

        if st.button("Send balloons!"):
            st.balloons()

    def load_heading(self):
        st.title("LLM CHATBOT")

    def load_chat_interface(self):
        # 🟢 1. Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 🟢 2. Sync self.state with session state
        self.state["messages"] = st.session_state.messages

        # 🟢 3. Display all previous messages
        for message in self.state["messages"]:
            if isinstance(message, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)

        # 🟢 4. Capture new input
        prompt = st.chat_input("Ask something...")

        if prompt:
            human_msg = HumanMessage(content=prompt)
            self.state["messages"].append(human_msg)
            st.session_state.messages = self.state["messages"]

            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                graph_builder = GraphBuilder(model=os.getenv("MODEL"), api=os.getenv("GROQ_API_KEY"))
                graph = graph_builder.chatbot_graph_builder()
                updated_state = graph.invoke(self.state)
                
                st.session_state.messages = updated_state["messages"]
                st.rerun()
            except Exception as e:
                with st.chat_message("ai"):
                    st.markdown(f"⚠️ Error: {e}")
