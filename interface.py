import streamlit as st
import json
from main import DocumentQA  # Assuming your class is in document_qa.py

# Initialize the DocumentQA system
documents_directory = './Documents/'
model_name = "llama3-8b-8192"
groq_api_key = "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"

qa_system = DocumentQA(documents_directory, model_name, groq_api_key)

# Streamlit interface
st.title("Document QA Chatbot")
st.write("Ask your questions about the documents!")

user_input = st.text_input("Your question:")

if st.button("Submit"):
    if user_input:
        # Get the response from the DocumentQA system
        json_result = qa_system.invoke_chain(user_input)

        # Parse the JSON result
        response_data = json.loads(json_result)  # Ensure the result is a Python dict

        # Display the answer separately
        st.subheader("Answer:")
        st.write(response_data.get("answer"))

        # Display the answer separately
        st.subheader("Source:")
        st.write(response_data.get("source_filename"))

        # Display the JSON result
        st.json(json_result)
    else:
        st.warning("Please enter a question.")
