import streamlit as st
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4")

# Streamlit UI for file upload
st.title("Chatbot with RAG")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Function to save the uploaded file locally
def save_uploaded_file(uploaded_file):
    with open(os.path.join("temp_uploaded_file.pdf"), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return "temp_uploaded_file.pdf"

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)  # Save and get file path
    st.success("File uploaded and saved successfully!")

    # Load the PDF and process it
    loader = PyPDFLoader(file_path)  # Load from saved file path
    document = loader.load()

    # Split the document into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    
    # Create embeddings and vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

else:
    st.warning("No document uploaded. The chatbot will work without RAG.")

# Input field for querying
user_query = st.text_input("Ask a question")

if user_query:
    if uploaded_file:
        st.write("Answer with RAG:")
        st.write(rag_chain.invoke(user_query))
    else:
        st.write("Answer without RAG:")
        st.write(llm.invoke([user_query]).content)
