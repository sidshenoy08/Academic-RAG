# Introduction

ContextLens is a RAG-powered application that has a knowledge base of research papers. 
It can answer user questions, providing title citations when possible and confidence levels.

The frontend is built using React, and the backend APIs are developed using FastAPI.


## Knowledge Base and Retrieval

The all-MiniLM-L6-v2 model is used to embed documents, while the chunk size and overlap size can be configured by the user when uploading a PDF document.
The vector embeddings and the corresponding chunks (with the paper title appended) are stored in ChromaDB.

The user question is embedded by the embedding model, and semantic search is performed to find the relevant information.
The top 5 relevant results are then passed as context to the LLM for response generation.

## Response Generation

Using the Gemini API, the original user question, along with the retrieved context and a prompt instructing it to refer only to the knowledge base at hand
is passed to the Gemini LLM.

The LLM is specifically instructed not to refer to any external source and will respond that it does not know the answer if the knowledge base lacks
relevant information.

## Application Features

The application includes user authentication using JWT tokens, creation, grouping, and deletion of chats, and chat history (similar to ChatGPT).

As the user can also like or dislike the model's responses, the application provides an analytical dashboard for the user to observe various metrics
such as the total number of likes/dislikes. The dashboard also allows the user to switch between monthly and yearly statistics.

The dashboard charts are built using D3.js.

User information and chat history are stored in MongoDB.


## Run the Application

Create a virtual environment

```
python -m venv .venv
```

Install the dependencies

```
pip install -r requirements.txt
```

Run the server

```
fastapi dev pipeline.py
```

The frontend is available at [ContextLens - Frontend](https://github.com/sidshenoy08/Academic-RAG-Frontend).

Install the dependencies

```
npm install
```

Instructions to run the frontend application are available in the repository.
