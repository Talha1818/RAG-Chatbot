# RAG-Chatbot

## Overview

This project is a Question Answering System built using RAG (Retrieval-Augmented Generation) architecture and LangChain. The application is designed to extract relevant information from PDFs and provide accurate answers to user queries. It combines powerful Natural Language Processing (NLP) models with an efficient document retrieval mechanism to deliver precise and context-aware responses.

## Installation

To run this project locally:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Talha1818/RAG-Chatbot.git
    cd RAG-Chatbot
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv env
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source env/bin/activate
        ```

4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the streamlit development server**:
    ```bash
    streamlit run interface.py
    ```
