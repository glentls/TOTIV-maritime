import streamlit as st
import openai
import os

client = openai(
    api_key=os.environ.get("OPENAI_API_KEY"), 
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
)

st.sidebar.image(image="/Users/yixin/Downloads/Logo (1).png")

st.sidebar.markdown("<h2 style='text-align: left; color: violet; '>Instructions!</h2",unsafe_allow_html=True)

st.sidebar.markdown("<h5 style='text-align: left; color: black; '>Key in the deficiency description into the text box and the Chatbot will provide you with the severity of the deficiency.</h5",unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: violet;'>TOTIV Chatypie</h1>",unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: black;'>How can I help today? :)</h5>",unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input(label="", key="input", placeholder= "Type your message here...")

if user_input: 
    st.session_state.messages.append({"role": "user", "content": user_input})

    chatbot_response = chat_completion(user_input)

    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

for message in st.session_state.messages:
    role = "User" if message["role"] == "user" else "Chatbot"
    st.markdown(f"**{role}:** {message['content']}")