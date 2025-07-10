from dotenv import load_dotenv
load_dotenv() 

import streamlit as st
from rag import graph
from os import getenv
OLLAMA_API_URL = getenv("CHAT_SERVER_URL")
MODEL_NAME = getenv("CHAT_MODEL")

st.set_page_config(page_title=getenv("APP_TITLE"), layout="centered")
st.title(getenv("APP_TITLE"))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("How may I assist you?")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare payload
    payload = {
        "model": MODEL_NAME,
        "messages": st.session_state.messages,
        "stream": True
    }

    # Stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = ""
            response_placeholder = st.empty()
            try:
                response = graph.invoke({"question": user_input})
                full_response = response['answer']
                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Connection error : {e}"
                response_placeholder.markdown(full_response)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
