import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from litellm import completion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hashlib import sha256
from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile, status, HTTPException, Response, Request
from pydantic import BaseModel, Field, AwareDatetime
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Optional, List
from io import BytesIO
from pymongo import MongoClient
from pydantic_mongo import PydanticObjectId, AbstractRepository
from datetime import datetime, timedelta, timezone
import jwt
import json
from bson import ObjectId

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


class User(BaseModel):
    id: Optional[PydanticObjectId] = None
    email: str
    password: str
    cpassword: str | None = Field(default=None, exclude=True)


class Message(BaseModel):
    userPrompt: str
    queryResponse: str


class Chat(BaseModel):
    id: Optional[PydanticObjectId] = None
    userId: Optional[PydanticObjectId] = None
    messages: List[Message]
    total_prompts: int = 1
    likes: int = 0
    dislikes: int = 0
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_updated: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class UserRepository(AbstractRepository[User]):
    class Meta:
        collection_name = 'users'


class ChatRepository(AbstractRepository[Chat]):
    class Meta:
        collection_name = 'chats'


client = chromadb.PersistentClient(path="./db")
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

gemini_api_key = os.getenv('GEMINI_API_KEY')

mongo_client = MongoClient(os.getenv('MONGODB_URL'), tz_aware=True)
database = mongo_client['academic-rag']
user_repo = UserRepository(database)
chat_repo = ChatRepository(database)


def issue_jwt_token(user_id):
    now = datetime.now(timezone.utc)
    expire_time = now + timedelta(minutes=60)
    payload = {
        'userId': user_id,
        'iat': int(now.timestamp()),
        'exp': int(expire_time.timestamp())
    }
    token = jwt.encode(payload, os.getenv('JWT_SECRET'), algorithm='HS256')
    return token


def extract_text_from_pdf(pdf_stream, chunk_size=512, chunk_overlap=50):
    all_text = ""
    reader = PdfReader(pdf_stream)
    for page in reader.pages:
        all_text += page.extract_text() or ""
    process_text_and_store(all_text, reader.metadata, chunk_size, chunk_overlap)
    # return all_text, reader.metadata


def process_text_and_store(all_text, metadata, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""], length_function=len
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


@app.post("/register")
async def register_user(user: User, response: Response):
    user.password = sha256(user.password.encode('utf-8')).hexdigest()
    result = user_repo.save(user)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    else:
        response.status_code = status.HTTP_201_CREATED
        token = issue_jwt_token(str(result.inserted_id))
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=True,
            samesite='none'
        )
        print("User has been registered!")


@app.post("/login")
async def login_user(user: User, response: Response):
    user = user_repo.find_one_by({'email': user.email, 'password': sha256(user.password.encode('utf-8')).hexdigest()})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    else:
        response.status_code = status.HTTP_200_OK
        token = issue_jwt_token(str(user.id))
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=True,
            samesite='none',
            path='/'
        )
        print("User has been logged in!")


@app.post("/submit")
async def submit_prompt(user_question: Annotated[str, Form()], chunk_size: Annotated[int, Form()], chunk_overlap: Annotated[int, Form()], files: list[UploadFile] | None = File(None)):
    # if files are uploaded
    if files:
        for file in files:
            pdf_bytes = await file.read()
            pdf_stream = BytesIO(pdf_bytes)
            extract_text_from_pdf(pdf_stream, chunk_size, chunk_overlap)
    knowledge_collection = client.get_or_create_collection(name="knowledgeBase")
    results = semantic_search(query=user_question, collection=knowledge_collection)
    context = "\n".join(results['documents'][0])
    response = generate_response(query=user_question, context=context)
    return {
        "model_response": response
    }


@app.post("/save")
async def save_chat(request: Request, response: Response):
    session_token = request.cookies.get('session_token')
    token = jwt.decode(session_token, os.getenv('JWT_SECRET'), algorithms=['HS256'])
    user_id = token.get('userId')
    request_body = await request.body()
    request_body_json = json.loads(request_body.decode('utf-8'))
    # if a new chat has been created by the user
    if not request_body_json.get('chatId'):
        message = Message(userPrompt=request_body_json.get('userPrompt'), queryResponse=request_body_json.get('queryResponse'))
        chat = Chat(userId=user_id, messages=[message])
        result = chat_repo.save(chat)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        else:
            response.status_code = status.HTTP_201_CREATED
        print("Current chat has been created!")
        return {"message": str(result.inserted_id)}
    else:
        chats_collection = database['chats']
        current_time = datetime.now(tz=timezone.utc)
        result = chats_collection.update_one(
            {"_id": ObjectId(request_body_json.get('chatId'))},
            {
                "$push": {
                    "messages": {"userPrompt": request_body_json.get('userPrompt'), "queryResponse": request_body_json.get('queryResponse')
                        }
                    },
                "$inc": {
                    "total_prompts": 1
                },
                "$set": {
                    "last_updated": current_time
                }
            },
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        else:
            response.status_code = status.HTTP_201_CREATED
        print("Current chat has been updated!")
        return {"message": str(result.upserted_id)}


@app.get("/retrieve")
async def retrieve_chats(request: Request, response: Response):
    session_token = request.cookies.get('session_token')
    token = jwt.decode(session_token, os.getenv('JWT_SECRET'), algorithms=['HS256'])
    user_id = ObjectId(token.get('userId'))
    user_chats = list(chat_repo.find_by({"userId": user_id}))
    if user_chats:
        response.status_code = status.HTTP_200_OK
        return {"userChats": user_chats}
    else:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"userChats": []}


@app.post("/like")
async def like_response(request: Request, response: Response):
    chats_collection = database['chats']
    request_body = await request.body()
    chat_id = json.loads(request_body.decode('utf-8')).get('chat')
    result = chats_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {
            "$inc": {
                "likes": 1
            }
        }
    )
    if result.modified_count > 0:
        print("Model response was liked!")
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_404_NOT_FOUND


@app.post("/dislike")
async def like_response(request: Request, response: Response):
    chats_collection = database['chats']
    request_body = await request.body()
    chat_id = json.loads(request_body.decode('utf-8')).get('chat')
    result = chats_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {
            "$inc": {
                "dislikes": 1
            }
        }
    )
    if result.modified_count > 0:
        print("Model response was disliked!")
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_404_NOT_FOUND


@app.post("/delete")
async def delete_chat(request: Request, response: Response):
    request_body = await request.body()
    chat_id = json.loads(request_body.decode('utf-8')).get('chat')
    result = chat_repo.delete_by_id(ObjectId(chat_id))
    if result.deleted_count > 0:
        print("Chat deleted successfully!")
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_404_NOT_FOUND


@app.post("/logout")
async def logout_user(request: Request, response: Response):
    # response.delete_cookie(key="session_token")
    response.status_code = status.HTTP_200_OK
    return {"message": "User has been logged out"}