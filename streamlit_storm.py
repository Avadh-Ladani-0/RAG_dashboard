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
import re

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define paths for Chroma and data storage
CHROMA_PATH = "chroma//chroma_research"
DATA_PATH = "data//research"

# Prompt templates for different use cases
PROMPTS = {
    "Section Content Generation": """
Generate content for the following section of a research paper. The content must align with the provided context and motive of the paper. Do not include irrelevant information or make assumptions outside the context.

Section: {section}

Motive: {motive}

Context:
{context}

---

Generate detailed and precise content for the section "{section}" that aligns with the motive.
""",

    "Summarization for Section": """
Summarize the relevant context to provide concise and focused content for the following section of a research paper. Ensure the summary aligns with the motive of the paper.

Section: {section}

Motive: {motive}

Context:
{context}

---

Provide a well-structured summary for the section "{section}".
""",

    "Detailed Explanation for Section": """
Provide a detailed explanation for the following section of a research paper. Ensure that the content is comprehensive and aligns with the motive provided. Avoid including unrelated details.

Section: {section}

Motive: {motive}

Context:
{context}

---

Generate an in-depth explanation for the section "{section}" based on the context and motive.
"""
}

# Initialize Streamlit app
st.title("Research Paper Content Generator")

# Sidebar for selecting embedding and chat models
st.sidebar.header("Settings")
embedding_option = st.sidebar.selectbox("Select Embedding Model", ["OpenAI Embeddings", "Hugging Face Embeddings"])
chat_model_option = st.sidebar.selectbox("Select Chat Model", ["GPT-3.5 Turbo", "LLaMA 3.1", "Mistral", "GPT-4"])
chunking_option = st.sidebar.selectbox("Select Chunking Method", ["Recursive Character Splitter", "Semantic Chunker"])
temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.4)
k_chunks = st.sidebar.slider("Number of Chunks in Similarity Search (k)", 1, 50, 10)

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload Research Paper PDFs", type="pdf", accept_multiple_files=True)
generate_db = st.button("Generate Vector Database")

# Input for outline and motive
st.write("### Provide Paper Details")
outline_text = st.text_area("Enter the outline (e.g., Introduction, Methods, Conclusion):", height=150)
motive = st.text_input("Enter the motive of the paper:")

# Helper functions
def extract_sections(text):
    """Extract sections from the text based on common research paper structure."""
    sections = {}
    current_section = None
    for line in text.split("\n"):
        if re.match(r"^(Introduction|Methods|Results|Discussion|Conclusion)", line, re.IGNORECASE):
            current_section = line.strip()
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line.strip())
    return {section: " ".join(content) for section, content in sections.items()}

def load_documents(uploaded_files):
    # Save uploaded PDFs to the DATA_PATH directory
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    for uploaded_file in uploaded_files:
        with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    sectioned_documents = []

    for doc in documents:
        sections = extract_sections(doc.page_content)
        for section, content in sections.items():
            metadata = doc.metadata.copy()
            metadata["section"] = section
            sectioned_documents.append(Document(page_content=content, metadata=metadata))

    return sectioned_documents

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

# Generate content for each section of the outline
if st.button("Generate Paper Content") and outline_text and motive:
    # Initialize vector database
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") if embedding_option == "Hugging Face Embeddings" else OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Process the outline
    outline_sections = [section.strip() for section in outline_text.split("\n") if section.strip()]
    st.write("### Generated Content")

    for section in outline_sections:
        query_text = f"Find similar sections to '{section}'"
        
        # Perform similarity search to find relevant sections
        results = db.similarity_search_with_relevance_scores(query_text, k=k_chunks)
        similar_sections = [doc for doc, _score in results if section.lower() in doc.metadata.get("section", "").lower()]

        # Collect all chunks from similar sections
        all_chunks = [doc.page_content for doc in similar_sections]
        # if not all_chunks:
        #     st.write(f"**{section}**: Unable to find relevant sections.")
        #     continue
        print(len(all_chunks))
        # Combine all chunks into a single context
        context_text = "\n\n---\n\n".join(all_chunks)
        prompt_template = ChatPromptTemplate.from_template(PROMPTS["Section Content Generation"])
        prompt = prompt_template.format(context=context_text, section=section, motive=motive)

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
        st.write(f"**{section}**:")
        st.write(response_text)
