import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from litellm import completion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hashlib import sha256
from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from io import BytesIO

# folder = os.fsencode(os.getenv('DIR_PATH'))

load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3000/home"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    user_question: str

client = chromadb.PersistentClient(path="./db")
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

gemini_api_key = os.getenv('GEMINI_API_KEY')


def extract_text_from_pdf(pdf_stream):
    all_text = ""
    reader = PdfReader(pdf_stream)
    for page in reader.pages:
        all_text += page.extract_text() or ""
    process_text_and_store(all_text, reader.metadata)
    # return all_text, reader.metadata


def process_text_and_store(all_text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""], length_function=len
    )

    hash_value = sha256(all_text.encode('utf-8')).hexdigest()
    chunks = text_splitter.split_text(all_text)

    knowledge_collection = client.get_or_create_collection(name="knowledgeBase")
    metadata_collection = client.get_or_create_collection(name="metadata", embedding_function=None)

    # check if paper has already been added to knowledge base
    results = metadata_collection.get(ids=[hash_value])

    if results["ids"]:
        print("Paper already exists in knowledge base")
        return

    title = metadata.title if metadata.title else ""
    publication_id = metadata["/IEEE Publication ID"] if "/IEEE Publication ID" in metadata.keys() else ""
    issue_id = metadata["/IEEE Issue ID"] if "/IEEE Issue ID" in metadata.keys() else ""
    article_id = metadata["/IEEE Article ID"] if "/IEEE Article ID" in metadata.keys() else ""

    metadata_collection.add(
        ids=[hash_value],
        documents=[hash_value],
        metadatas=[{"title": title, "publication_id": publication_id, "issue_id": issue_id, "article_id": article_id}]
    )

    for i, chunk in enumerate(chunks):
        embedding = text_embedding_model.encode(chunk)
        knowledge_collection.add(
            ids=[f"{hash_value}_chunk_{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"source": hash_value, "chunk_id": i}],
            documents=[f'{chunk}\nTitle: {title}']
        )
    print("File uploaded to knowledge base!")

def semantic_search(query, collection, top_k=5):
    query_embedding = text_embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k
    )
    return results


def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {context}"
    response = completion(
        model="gemini/gemini-3.5-flash",
        messages=[{"content": "You are an academic expert who has read all the papers provided in the context. Answer the given query based on your knowledge from these papers ONLY. Do NOT use external resources. Cite the paper title based on the context provided in a readable format. The title is at the end of each text chunk with the key 'Title:'. Provide a percentage of how confident you are of your answer.", "role": "system"}, {"content": prompt, "role": "user"}],
        api_key=gemini_api_key
    )
    return response['choices'][0]['message']['content']


def get_source_metadata(sources, collection):
    source_ids = list({
        paper['source']
        for paper in sources['metadatas'][0]
    })
    source_metadata = collection.get(ids=source_ids)
    return source_metadata


# for file in os.listdir(folder):
#     filename = os.fsdecode(file)
#     pdf_text, pdf_metadata = extract_text_from_pdf(filename)
#     process_text_and_store(all_text=pdf_text, metadata=pdf_metadata)


# knowledge_collection = client.get_or_create_collection(name="knowledgeBase")
# results = knowledge_collection.get()
#
# print(results)

# knowledge_collection = client.get_or_create_collection(name="knowledgeBase")
# # metadata_collection = client.get_or_create_collection(name="metadata", embedding_function=None)
# query = input("Please ask a question about UAVs: ")
# results = semantic_search(query, knowledge_collection)
# # # source_metadata = get_source_metadata(results, metadata_collection)
# context = "\n".join(results['documents'][0])
# # # source_context = "\n".join(source_metadata)
# response = generate_response(query, context)
# print(response)


# @app.post("/submit")
# def submit_prompt(prompt: Prompt):
#     # when no file is uploaded
#     knowledge_collection = client.get_or_create_collection(name="knowledgeBase")
#     results = semantic_search(query=prompt.user_question, collection=knowledge_collection)
#     context = "\n".join(results['documents'][0])
#     response = generate_response(query=prompt.user_question, context=context)
#     return {
#         "model_response": response
#     }

@app.post("/submit")
async def submit_prompt(user_question: Annotated[str, Form()], files: list[UploadFile] | None = File(None)):
    # if files are uploaded
    if files:
        for file in files:
            pdf_bytes = await file.read()
            pdf_stream = BytesIO(pdf_bytes)
            extract_text_from_pdf(pdf_stream)
    knowledge_collection = client.get_or_create_collection(name="knowledgeBase")
    results = semantic_search(query=user_question, collection=knowledge_collection)
    context = "\n".join(results['documents'][0])
    response = generate_response(query=user_question, context=context)
    return {
        "model_response": response
    }