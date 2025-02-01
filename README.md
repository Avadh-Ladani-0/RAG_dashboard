# RAG Dash: Retrieval-Augmented Generation Framework

## Overview
This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document-based question answering. The core functionality is encapsulated in the `RAG_dash.py` script, which integrates document processing, embeddings comparison, and query handling into a seamless pipeline.

### Key Features:
- **Document Preprocessing**: Load and preprocess PDF documents, chunking text for embedding storage.
- **Embeddings Management**: Compare document embeddings for precise retrieval of relevant information.
- **Query Handling**: Retrieve context-specific answers from documents using OpenAI embeddings.
- **Streamlit Dashboard**: A user-friendly interface to interact with the RAG pipeline.

## Folder Structure
```
├── .gitignore              # Files and directories to ignore
├── RAG_dash.py             # Main application script with the integrated RAG pipeline
├── create_database.py      # Script for generating embeddings and storing them in a vector database
├── compare_embeddings.py   # Script for comparing document embeddings
├── query_data.py           # Script for querying the vector database
├── streamlit_game.py       # Additional streamlit demo application (beta)
├── streamlit_storm.py      # Additional streamlit demo application (beta)
├── requirements.txt        # Required dependencies
├── README.md               # Project documentation
```

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Avadh-Ladani-0/RAG_dashboard.git
   cd RAG_dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Obtain your API key from the [OpenAI platform](https://platform.openai.com/) or for llama/mistral setup download and setup ollama from ollama download llama/mistral.
   - save your openai api key on .env 
   - Add it to your environment variables or directly into the scripts (not recommended for production).

## Usage

### Running the Main Application
To launch the Streamlit dashboard:
```bash
streamlit run RAG_dash.py
```

This will open a browser window where you can:
- Upload PDF documents.
- Query the documents for answers.
- View source references for each answer.

### Additional Scripts
- **`create_database.py`**: Preprocess PDFs and save their embeddings to a vector database.
- **`compare_embeddings.py`**: Compare embeddings to identify the most relevant chunks for a given query.
- **`query_data.py`**: Query the vector database for answers and fetch relevant documents.

### Demo Applications
You can also explore the demo applications:
- `streamlit_game.py`
- `streamlit_storm.py`

Run them using:
```bash
streamlit run <filename>.py
```

## Project Flow
1. **Document Processing**: Upload PDFs, split them into chunks, and generate embeddings using `create_database.py`.
2. **Querying**: Use `RAG_dash.py` to input a query and retrieve results, complete with source references.

## Contributing
We welcome contributions! If you'd like to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **OpenAI** for their embeddings API.
- **LangChain** and **Chroma** for facilitating document management and vector database integration.

---
For any questions or issues, feel free to create an issue or reach out to me at [avadhladani2002@gmail.com].
