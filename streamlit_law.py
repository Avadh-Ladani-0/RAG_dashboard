import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define paths for Chroma and data storage
CHROMA_PATH = "chroma1//chroma_law"
DATA_PATH = "data1//law"

# Prompt templates for different use cases
PROMPTS = {
    "General Information": """"
Answer the question based only on the following context don't try to makeup the answer:

{context}

---

Answer the question in detail (max 2 paragraphs) only if needed otherwise answer the question in small to medium size based on the above context: {question}
""",

    "Summarization": "Summarize the content based on the following :\n\n{context}\n\n---\n\n{question}",
    "Detailed Explanation": "Give a detailed explanation based only on the following context:\n\n{context}\n\n---\n\n{question}"
}

# Initialize Streamlit app
st.title("Enhanced RAG QA : Chatbot for the law domain")

# Sidebar for selecting embedding and chat models
st.sidebar.header("Settings")
embedding_option = st.sidebar.selectbox("Select Embedding Model", ["OpenAI Embeddings", "Hugging Face Embeddings"])
chat_model_option = st.sidebar.selectbox("Select Chat Model", ["GPT-3.5 Turbo", "LLaMA 3.1", "Mistral", "GPT-4"])
chunking_option = st.sidebar.selectbox("Select Chunking Method", ["Recursive Character Splitter", "Semantic Chunker"])
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.4)
k_chunks = st.sidebar.slider("Number of Chunks in Similarity Search (k)", 1, 50, 10)
prompt_option = st.sidebar.selectbox("Select Prompt Template", list(PROMPTS.keys()))

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
generate_db = st.button("Generate Vector Database")

# Helper functions
def load_documents(uploaded_files):
    # Save uploaded PDFs to the DATA_PATH directory
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    for uploaded_file in uploaded_files:
        with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_text(documents):
    # Choose the embedding model for SemanticChunker
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") if embedding_option == "Hugging Face Embeddings" else OpenAIEmbeddings()
    
    # Choose the chunking method
    if chunking_option == "Recursive Character Splitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
    else:
        text_splitter = SemanticChunker(embedding_function, breakpoint_threshold_type="percentile")
    
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    # Clear out existing database if needed
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") if embedding_option == "Hugging Face Embeddings" else OpenAIEmbeddings()
    
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    db.persist()
    st.write("Vector database created successfully.")

# Generate vector database when the button is clicked
if generate_db and uploaded_files:
    documents = load_documents(uploaded_files)
    chunks = split_text(documents)
    save_to_chroma(chunks)

# Query section
query_text = st.text_input("Enter your query:")
if st.button("Get Answer") and query_text:
    # Initialize vector database
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") if embedding_option == "Hugging Face Embeddings" else OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=k_chunks)
    if len(results) == 0 or results[0][1] < 0.5:
        st.write("Unable to find matching results.")
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPTS[prompt_option])
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Choose the chat model
        if chat_model_option == "GPT-3.5 Turbo":
            model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
        elif chat_model_option == "LLaMA 3.1":
            model = OllamaLLM(model="llama3.1", temperature=temperature)
        elif chat_model_option == "Mistral":
            model = OllamaLLM(model="mistral", temperature=temperature)
        else:  # GPT-4
            model = ChatOpenAI(model_name="gpt-4", temperature=temperature)

        # Get the response from the selected model
        response_text = model.invoke(prompt)
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        st.write("**Response:**")
        st.write(response_text)
        st.write("**Sources:**")
        st.write(sources)
