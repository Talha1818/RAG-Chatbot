import streamlit as st
import json
from main import DocumentQA  # Assuming your class is in main.py

# Initialize the DocumentQA system
documents_directory = './Documents/'
model_name = "llama3-8b-8192"
groq_api_key = "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"

qa_system = DocumentQA(documents_directory, model_name, groq_api_key)

# Streamlit interface
st.title("Document QA Chatbot")
st.write("Ask your questions about the documents!")

# Initialize session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['user']}")
    st.write(f"**Bot:** {entry['bot']}")

# User input
user_input = st.text_input("Your question:")

if st.button("Submit"):
    if user_input:
        # Get the response from the DocumentQA system
        json_result = qa_system.invoke_chain(user_input)

        # Parse the JSON result
        response_data = json.loads(json_result)  # Ensure the result is a Python dict

        # Store the question and answer in chat history
        st.session_state.chat_history.append({
            'user': user_input,
            'bot': response_data.get("answer", "I don't know.")
        })

        # Clear the input box (optional) by creating a new text_input with an empty default
        st.text_input("Your question:", value='', key='new_input')
    else:
        st.warning("Please enter a question.")
