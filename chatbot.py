import streamlit as st

st.title(":blue[TOTIV Chatypie]")
st.markdown("Ask me something! :)")
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", key="input", placeholder= "Type your message here...")