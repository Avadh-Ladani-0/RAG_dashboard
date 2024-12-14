# Import necessary modules
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import openai
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Define paths for Chroma and data storage
CHROMA_PATH = r"langchain-rag-tutorial-main\\chroma"
DATA_PATH = r"langchain-rag-tutorial-main\\data\\law"

def load_documents():
    # Load all PDFs in the specified directory
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter1 = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    text_splitter= SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),breakpoint_threshold_type="interquartile"
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    # Use HuggingFace LLaMA embeddings
    embeddings1 = OpenAIEmbeddings()
    embeddings2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embeddings2, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
