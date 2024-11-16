import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class DocumentQA:
    def __init__(self, documents_directory, model_name, groq_api_key):
        # Load modules and prepare necessary components once
        self.loader = DirectoryLoader(documents_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
        self.pages = self.loader.load()

        self.instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_documents(self.pages)
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=self.splits, embedding=self.instructor_embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        self.llm = ChatGroq(model=model_name, temperature=0, groq_api_key=groq_api_key)

        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

    def format_results_as_json(self, results):
        formatted_results = {
            "input": results.get("input"),
            "context": [],
            "answer": results.get("answer"),
            "source_docs": [],
            "source_filename": ""
        }

        source_docs = []

        # Extracting relevant document information
        for document in results.get("context", []):
            doc_info = {
                "id": document.id,
                "metadata": document.metadata,
                "page_content": document.page_content
            }

            source_docs.append({
                "File": document.metadata['source'].split("/")[-1],
                "Page#": document.metadata['page']
            })

            formatted_results["context"].append(doc_info)

        formatted_results["source_docs"].append(source_docs)
        formatted_results["source_filename"] = (f"\nNote: \n(These answers were generated from these files: {source_docs})")

        return json.dumps(formatted_results, indent=4)

    def invoke_chain(self, user_input):
        response = self.rag_chain.invoke({"input": user_input})
        results = {
            'input': response['input'],
            'context': response['context'],
            'answer': response['answer']
        }

        # Convert results to JSON format
        json_output = self.format_results_as_json(results)
        return json_output


if __name__ == "__main__":
    documents_directory = './Documents/'
    model_name = "llama3-8b-8192"
    groq_api_key = "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"
    
    qa_system = DocumentQA(documents_directory, model_name, groq_api_key)
    user_query = "Unacceptable Use?"
    json_result = qa_system.invoke_chain(user_query)
    print(json_result)
