
import streamlit as st
import sys
import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import your existing components
from app.agents import app as agent_app

st.set_page_config(
    page_title="🧳 Travel Chatbot",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChat [data-testid="stChatMessageContent"] {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    
    .main-header {
        text-align: center;
        color: #1976d2;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"messages": []}


with st.sidebar:
    st.header("🧳 Travel Assistant")
    st.markdown("---")
    
    # Model info
    st.subheader("🤖 Model Information")
    st.info("Using Google Gemini 1.5 Flash for intelligent responses")
    
    # Features
    st.subheader("✨ Features")
    st.markdown("""
    - 📚 **PDF Search**: Search through travel documents
    - 🌐 **Web Search**: Get real-time travel information
    - 🎯 **Smart Routing**: Automatically chooses best information source
    - 💬 **Context Aware**: Maintains conversation context
    """)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_state = {"messages": []}
        st.rerun()
    
    # Statistics
    st.subheader("📊 Chat Statistics")
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", total_messages)
    with col2:
        st.metric("Q&A Pairs", user_messages)

# Main content
st.markdown("<h1 class='main-header'>🧳 Travel Chatbot Assistant</h1>", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class='info-box'>
    <strong>🎯 How to use:</strong> Ask me anything about travel! I can search through your travel documents 
    and provide real-time information from the web. Try questions like:
    <ul>
        <li>"What places to visit in Manali?"</li>
        <li>"Best time to visit Goa?"</li>
        <li>"Travel requirements for Dubai?"</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about travel..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching for information..."):
                try:
                    # Add user message to agent state
                    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))
                    
                    # Get response from agent
                    response = agent_app.invoke(st.session_state.agent_state)
                    
                    # Extract the latest AI message
                    latest_message = response["messages"][-1]
                    if hasattr(latest_message, 'content'):
                        bot_response = latest_message.content
                    else:
                        bot_response = str(latest_message)
                    
                    # Update agent state
                    st.session_state.agent_state = response
                    
                    # Display response
                    st.markdown(bot_response)
                    
                    # Add bot response to chat
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    
                except Exception as e:
                    error_msg = f"🚨 Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🧳 Travel Chatbot powered by Google Gemini & LangGraph"
    "</div>", 
    unsafe_allow_html=True
)