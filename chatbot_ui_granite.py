import os
import dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import VLLMOpenAI

# Load environment variables
dotenv.load_dotenv()

# Setup for VLLM LLM instance
llm = VLLMOpenAI(
    openai_api_key="EMPTY",  # API key set to EMPTY as vLLM doesn't require one
    openai_api_base="http://localhost:8000/v1",  # Your local vLLM endpoint
    model_name="ibm/granite-7b-instruct",  # Adjust the model name as needed
    model_kwargs={"stop": ["."]}
)

# Streamlit UI for file upload
st.set_page_config(layout="wide")
st.title("Chatbot with RAG using vLLM")

# Layout for left and right columns
left_col, right_col = st.columns(2)

# File upload on the left column
with left_col:
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    

# Function to save the uploaded file locally
def save_uploaded_file(uploaded_file):
    with open(os.path.join("temp_uploaded_file.pdf"), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_uploaded_file.pdf"

# Variables to store the vectorstore and retriever
vectorstore = None
retriever = None

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)  # Save and get file path
    with left_col:
        st.success("File uploaded and saved successfully!")

    # Load the PDF and process it
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)

    # Create embeddings using all-MiniLM-L6-v2
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()

    with left_col:
        st.success("PDF processed and embeddings created.")
else:
    with left_col:
        st.warning("No document uploaded. The chatbot will work without RAG.")

# Input field for querying, making it smaller for better UX
user_query = st.text_input("Ask a question", max_chars=150)

if user_query:
    if retriever:  # Check if the retriever is defined (i.e., document uploaded)
        # Retrieve relevant context using RAG
        context_documents = retriever.get_relevant_documents(user_query)
        context = "\n\n".join(doc.page_content for doc in context_documents)

        # Define a structured and enhanced prompt following the required format
        prompt = (
            f"<|system|>\n"
            "You are an AI language model developed by IBM Research. You are a cautious assistant. "
            "You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n\n"
            f"<|user|>\n"
            f"You are an AI language model designed to function as a specialized Retrieval Augmented Generation (RAG) assistant. "
            "When generating responses, prioritize correctness, i.e., ensure that your response is grounded in context and user query. "
            "Always make sure that your response is relevant to the question.\n\n"
            f"Answer length: narrative\n\n"
            f"[Document]\n\n"
            f"{file_path}\n\n"
            f"{context}\n\n"
            f"[End]\n\n"
            f"{user_query}\n\n"
            f"<|assistant|>"      
        )

        with left_col:
            st.write("Answer with RAG:")
        # Use vLLM with the prompt
        response = llm(prompt)

        with left_col:
            st.write(response)

        # Show context and prompt on the right column
        with right_col:
            #st.subheader("Fetched Context")
            #st.write(context)

            st.subheader("Generated Prompt")
            st.write(prompt)

    else:
        with left_col:
            st.write("Answer without RAG:")
        # Call vLLM API for user_query without RAG
        response = llm(user_query)

        with left_col:
            st.write(response)
