# #code2
# import os
# import uuid
# import logging
# import json
# import asyncio
# from datetime import datetime
# from typing import List, Dict, Any, AsyncGenerator, Optional

# from fastapi import FastAPI, Depends, HTTPException, status, Query
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel, Field
# from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text, func
# from sqlalchemy.orm import sessionmaker, Session, relationship
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.dialects.postgresql import JSONB
# from fastapi.middleware.cors import CORSMiddleware
# # --- New Imports for Language Model ---
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# # --- Basic Logging Configuration ---
# # This will print log messages to your console, including the time, log level, and message.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- Configuration ---
# # Your Azure PostgreSQL connection details

# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_NAME = os.getenv("DB_NAME")
# DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# # --- IMPORTANT: Replace with your actual API key and Base URL ---

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# BASE_URL = os.getenv("LLM_BASE_URL")
# MODEL_NAME = os.getenv("MODEL_NAME")
# INSURANCE_SAFETY_SETTINGS = {
#     "safety_settings": [
#         {
#             "category": "HARM_CATEGORY_HATE_SPEECH",
#             "threshold": "BLOCK_LOW_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#             "threshold": "BLOCK_LOW_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_HARASSMENT",
#             "threshold": "BLOCK_LOW_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#             "threshold": "BLOCK_LOW_AND_ABOVE"
#         }
#     ]
# }
# logging.info("Configuration loaded. Connecting to PostgreSQL database.")

# # --- Language Model Setup ---
# try:
#     llm = ChatOpenAI(
#         model_name=MODEL_NAME,
#         api_key=OPENAI_API_KEY,
#         base_url=BASE_URL,
#         streaming=True,
#     	extra_body=INSURANCE_SAFETY_SETTINGS
#         )
#     logging.info(f"Language model '{MODEL_NAME}' initialized successfully.")
# except Exception as e:
#     logging.error(f"Failed to initialize language model: {e}")
#     # You might want to exit here if the LLM is critical
#     # exit(1)

# # --- SQLAlchemy Setup ---
# engine = create_engine(
#     DATABASE_URL,
#     pool_pre_ping=True,  # Verify connections before using them
#     pool_recycle=3600,   # Recycle connections after 1 hour (3600 seconds)
#     pool_size=10,        # Number of persistent connections
#     max_overflow=20,     # Additional connections when pool is exhausted
#     connect_args={
#         "connect_timeout": 10,
#         "keepalives": 1,
#         "keepalives_idle": 30,
#         "keepalives_interval": 10,
#         "keepalives_count": 5,
#     }
# )
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()
# logging.info("SQLAlchemy engine and session created with connection pooling.")


# # --- SQLAlchemy ORM Models ---
# class User(Base):
#     __tablename__ = "users"
#     user_id = Column(String, primary_key=True, index=True)
#     customer_details = Column(JSONB)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     conversations = relationship("Conversation", back_populates="user")

# class Conversation(Base):
#     __tablename__ = "conversations"
#     conversation_id = Column(String, primary_key=True, index=True)
#     user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     user = relationship("User", back_populates="conversations")
#     messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

# class Message(Base):
#     __tablename__ = "messages"
#     message_id = Column(String, primary_key=True, index=True)
#     conversation_id = Column(String, ForeignKey("conversations.conversation_id"), nullable=False)
#     sender_id = Column(String, nullable=False)
#     message_text = Column(Text, nullable=False)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     conversation = relationship("Conversation", back_populates="messages")


# # --- Pydantic Models ---
# class CreateConversationRequest(BaseModel):
#     user_id: str
#     customer_details: Dict[str, Any]

# class MessageHistoryItem(BaseModel):
#     message_id: str
#     sender_id: str
#     message_text: str
#     created_at: datetime
#     class Config:
#         orm_mode = True

# class CreateConversationResponse(BaseModel):
#     conversation_id: str
#     initial_message: str
#     history: List[MessageHistoryItem]

# class SendMessageRequest(BaseModel):
#     conversation_id: str
#     sender_id: str
#     message_text: str

# # NOTE: SendMessageResponse is no longer used by the /messages endpoint,
# # but can be kept for other potential uses.
# class SendMessageResponse(BaseModel):
#     chatbot_response: str

# class PaginationInfo(BaseModel):
#     page: int
#     page_size: int
#     total_messages: int
#     total_pages: int

# class ChatHistoryResponse(BaseModel):
#     history: List[MessageHistoryItem]
#     pagination: PaginationInfo


# # --- FastAPI Application ---
# app = FastAPI(
#     title="Chat Service API",
#     description="APIs for creating conversations, sending messages (now with streaming), and fetching chat history.",
#     version="1.1.0"
# )
# app.add_middleware(
#     CORSMiddleware,
#     #allow_origins=["http://localhost:3001","https://ifli.digitalfd.net:9001", "https://fg-dsm.digitalfd.net:9004", "https://fg-dsm.digitalfd.net:9001", "https://fg-dsm.digitalfd.net:9002", "https://ifli.digitalfd.net:9002"],  
#     allow_origins= ["*"],
#     allow_credentials=False,
#     allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"],
#     expose_headers=["*"],  # Add this
#     max_age=3600,# Allows all headers
# )


# # @app.options("/{path:path}")
# # async def options_handler(path: str):
# #     """Handle all OPTIONS (preflight) requests"""
# #     return Response(
# #         status_code=200,
# #         headers={
# #             "Access-Control-Allow-Origin": "*",
# #             "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
# #             "Access-Control-Allow-Headers": "*",
# #             "Access-Control-Max-Age": "3600",
# #         }
# #     )
# # --- Dependency for DB session ---
# def get_db():
#     logging.info("Opening new database session.")
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         logging.info("Closing database session.")
#         db.close()


# # --- API Endpoints ---

# @app.post("/conversations", response_model=CreateConversationResponse, status_code=status.HTTP_201_CREATED, tags=["Conversation"])
# def create_conversation(req: CreateConversationRequest, db: Session = Depends(get_db)):
#     logging.info(f"Received request to create conversation for user_id: {req.user_id}")
    
#     # Check if user exists
#     db_user = db.query(User).filter(User.user_id == req.user_id).first()
#     if not db_user:
#         logging.info(f"User with user_id: {req.user_id} not found. Creating new user.")
#         db_user = User(user_id=req.user_id, customer_details=req.customer_details)
#         db.add(db_user)
#         db.commit()
#         db.refresh(db_user)
#     else:
#         logging.info(f"Found existing user with user_id: {req.user_id}")

#     # Check if conversation already exists for this user
#     existing_conversation = db.query(Conversation).filter(Conversation.user_id == req.user_id).first()
    
#     if existing_conversation:
#         logging.info(f"Found existing conversation {existing_conversation.conversation_id} for user {req.user_id}")
        
#         # Get all messages for this conversation
#         messages = db.query(Message).filter(
#             Message.conversation_id == existing_conversation.conversation_id
#         ).order_by(Message.created_at.asc()).all()
        
#         # Get the first chatbot message as initial_message
#         initial_message = next(
#             (msg.message_text for msg in messages if msg.sender_id == "chatbot"),
#             "Welcome back!"
#         )
        
#         logging.info(f"Returning existing conversation with {len(messages)} messages")
#         return {
#             "conversation_id": existing_conversation.conversation_id,
#             "initial_message": initial_message,
#             "history": messages
#         }
    
#     # Create new conversation if none exists
#     new_conversation_id = str(uuid.uuid4())
#     logging.info(f"Creating new conversation with conversation_id: {new_conversation_id}")
#     db_conversation = Conversation(conversation_id=new_conversation_id, user_id=req.user_id)
#     db.add(db_conversation)

#     try:
#         customer_name = req.customer_details.get("name", "there")
#         # initial_prompt = [SystemMessage(content=f"You are a friendly customer support agent. Start the conversation with a warm welcome to {customer_name}.")]

#         initial_prompt = [
#             {"role": "system", "content": "You are a friendly and professional customer support agent."},
#             {"role": "user", "content": f"Start the conversation with a warm, personalized welcome to {customer_name}."}
#         ]
        
#         logging.info("Invoking language model to generate a welcome message.")
#         initial_response = llm.invoke(initial_prompt)
#         welcome_message_text = getattr(initial_response, "content", None) or f"Welcome {customer_name}! How can I assist you today?"
#         # welcome_message_text = initial_response.content
#         logging.info("Successfully received welcome message from language model.")

#         welcome_message = Message(
#             message_id=str(uuid.uuid4()),
#             conversation_id=new_conversation_id,
#             sender_id="chatbot",
#             message_text=welcome_message_text
#         )
#         db.add(welcome_message)
#         db.commit()
#         db.refresh(db_conversation)
#         db.refresh(welcome_message)
#         logging.info(f"Conversation {new_conversation_id} and initial message saved to database.")

#     except Exception as e:
#         logging.error(f"Error invoking language model or saving initial message: {e}", exc_info=True)
#         db.rollback()
#         raise HTTPException(status_code=503, detail=f"Could not contact language model: {e}")

#     return {
#         "conversation_id": new_conversation_id,
#         "initial_message": welcome_message_text,
#         "history": [welcome_message]
#     }


# # --- NEW: Asynchronous Generator for Streaming Response ---
# async def stream_chatbot_response(
#     conversation_id: str,
#     messages_for_llm: List[SystemMessage | AIMessage | HumanMessage],
#     db: Session
# ) -> AsyncGenerator[str, None]:
#     """
#     This async generator streams the LLM response and saves the full message to the DB at the end.
#     """
#     full_response_text = ""
#     logging.info("Starting LLM stream for conversation.")
#     try:
#         # Use .astream() for asynchronous streaming
#         async for chunk in llm.astream(messages_for_llm):
#             # The chunk object has a `content` attribute with the text
#             content = chunk.content
#             if content:
#                 full_response_text += content
#                 # Format the chunk as a Server-Sent Event (SSE)
#                 # payload = {"chatbot_response_chunk": content}
#                 #yield f"data: {json.dumps(payload)}\n\n"
#                 yield content
#                 # Small sleep to allow the message to be sent
#                 await asyncio.sleep(0.01)

#     except Exception as e:
#         logging.error(f"Error during LLM stream for conversation {conversation_id}: {e}", exc_info=True)
#         error_payload = {"error": f"An error occurred with the language model service: {e}"}
#         yield f"data: {json.dumps(error_payload)}\n\n"
#         # Stop the generator and do not save to DB if an error occurs
#         return

#     # Once the stream is complete, save the full response to the database
#     if full_response_text:
#         logging.info(f"Stream complete for conversation {conversation_id}. Saving full response to database.")
#         try:
#             chatbot_message = Message(
#                 message_id=str(uuid.uuid4()),
#                 conversation_id=conversation_id,
#                 sender_id="chatbot",
#                 message_text=full_response_text
#             )
#             db.add(chatbot_message)
#             db.commit()
#             logging.info(f"Chatbot response for conversation {conversation_id} committed to the database.")
#         except Exception as e:
#             logging.error(f"Failed to save streamed response to DB for conversation {conversation_id}: {e}")
#             db.rollback()
#     else:
#         logging.warning(f"LLM stream for conversation {conversation_id} produced an empty response.")


# @app.post("/messages", tags=["Messaging"])
# async def send_message(req: SendMessageRequest, db: Session = Depends(get_db)): # MODIFIED to async def
#     logging.info(f"Received message from sender '{req.sender_id}' for conversation '{req.conversation_id}'")
    
#     conversation = db.query(Conversation).filter(Conversation.conversation_id == req.conversation_id).first()
#     if not conversation:
#         logging.warning(f"Conversation not found for conversation_id: {req.conversation_id}")
#         raise HTTPException(status_code=404, detail="Conversation not found")

#     logging.info("Saving user message to the database.")
#     user_message = Message(
#         message_id=str(uuid.uuid4()),
#         conversation_id=req.conversation_id,
#         sender_id=req.sender_id,
#         message_text=req.message_text
#     )
#     db.add(user_message)
#     db.commit()

#     logging.info(f"Fetching message history for conversation_id: {req.conversation_id} to build context.")
#     history = db.query(Message).filter(Message.conversation_id == req.conversation_id).order_by(Message.created_at.desc()).limit(10).all()
#     history.reverse() # Order from oldest to newest for the LLM
#     logging.info(f"Found {len(history)} messages for context.")

#     messages_for_llm = [SystemMessage(content="You are a helpful assistant.")]
#     for msg in history:
#         if msg.sender_id == 'chatbot':
#             messages_for_llm.append(AIMessage(content=msg.message_text))
#         else:
#             messages_for_llm.append(HumanMessage(content=msg.message_text))
    
#     # Create the generator and return it in a StreamingResponse
#     response_generator = stream_chatbot_response(req.conversation_id, messages_for_llm, db)
#     return StreamingResponse(response_generator, media_type="text/plain")


# @app.get("/conversations/{conversation_id}/history", response_model=ChatHistoryResponse, tags=["Conversation"])
# def get_chat_history(
#     conversation_id: str,
#     page: int = Query(1, ge=1, description="Page number (starting from 1)"),
#     page_size: int = Query(50, ge=1, le=200, description="Number of messages per page"),
#     db: Session = Depends(get_db)
# ):
#     logging.info(f"Request received for chat history of conversation_id: {conversation_id} (page: {page}, page_size: {page_size})")
    
#     conversation = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
#     if not conversation:
#         logging.warning(f"History requested for non-existent conversation_id: {conversation_id}")
#         raise HTTPException(status_code=404, detail="Conversation not found")

#     # Get total count of messages
#     total_messages = db.query(func.count(Message.message_id)).filter(
#         Message.conversation_id == conversation_id
#     ).scalar()
    
#     # Calculate pagination
#     total_pages = (total_messages + page_size - 1) // page_size  # Ceiling division
#     offset = (page - 1) * page_size
    
#     # Get paginated messages
#     messages = db.query(Message).filter(
#         Message.conversation_id == conversation_id
#     ).order_by(Message.created_at.asc()).offset(offset).limit(page_size).all()
    
#     logging.info(f"Found {len(messages)} messages for page {page} of conversation_id: {conversation_id}. Total messages: {total_messages}")
    
#     return {
#         "history": messages,
#         "pagination": {
#             "page": page,
#             "page_size": page_size,
#             "total_messages": total_messages,
#             "total_pages": total_pages
#         }
#     }


# @app.get("/health", tags=["Health"])
# async def health_check():
#     logging.info("Health check endpoint was called.")
#     return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
 
# if __name__ == "__main__":
#     import uvicorn
#     logging.info("Starting Uvicorn server.")
#     uvicorn.run(
#         "__main__:app",
#         host="0.0.0.0",
#         port=5000,
#         reload=True
#     )



#code2
import os
import uuid
import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Optional

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from fastapi.middleware.cors import CORSMiddleware
# --- New Imports for Language Model ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Basic Logging Configuration ---
# This will print log messages to your console, including the time, log level, and message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Your Azure PostgreSQL connection details

DB_USER = os.getenv("DB_USER", "fdocertool")
DB_PASSWORD = os.getenv("DB_PASSWORD", "!QAZ1qaz")
DB_HOST = os.getenv("DB_HOST", "fdocrtool.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "proxy_IFLI")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# --- IMPORTANT: Replace with your actual API key and Base URL ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk--wx6ZybhSrNenNuxjSmhNA")
BASE_URL = os.getenv("LLM_BASE_URL", "https://aura.dev.ryzeai.ai")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini/gemini-2.5-pro")

logging.info("Configuration loaded. Connecting to PostgreSQL database.")

# --- Language Model Setup ---



INSURANCE_SAFETY_SETTINGS = {
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        }
    ]
}

try:
    llm = ChatOpenAI(
            model="ifli/gemini-2.5-pro",
            base_url="https://aura.dev.ryzeai.ai",
            api_key="sk--wx6ZybhSrNenNuxjSmhNA",  
            temperature=0.2,
            extra_body=INSURANCE_SAFETY_SETTINGS
        )
    logging.info(f"Language model '{MODEL_NAME}' initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize language model: {e}")
    # You might want to exit here if the LLM is critical
    # exit(1)

# --- SQLAlchemy Setup ---
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour (3600 seconds)
    pool_size=10,        # Number of persistent connections
    max_overflow=20,     # Additional connections when pool is exhausted
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
logging.info("SQLAlchemy engine and session created with connection pooling.")


# --- SQLAlchemy ORM Models ---
class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True)
    customer_details = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"
    conversation_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    message_id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.conversation_id"), nullable=False)
    sender_id = Column(String, nullable=False)
    message_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    conversation = relationship("Conversation", back_populates="messages")


# --- Pydantic Models ---
class CreateConversationRequest(BaseModel):
    user_id: str
    customer_details: Dict[str, Any]

class MessageHistoryItem(BaseModel):
    message_id: str
    sender_id: str
    message_text: str
    created_at: datetime
    class Config:
        orm_mode = True

class CreateConversationResponse(BaseModel):
    conversation_id: str
    initial_message: str
    history: List[MessageHistoryItem]

class SendMessageRequest(BaseModel):
    conversation_id: str
    sender_id: str
    message_text: str

# NOTE: SendMessageResponse is no longer used by the /messages endpoint,
# but can be kept for other potential uses.
class SendMessageResponse(BaseModel):
    chatbot_response: str

class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total_messages: int
    total_pages: int

class ChatHistoryResponse(BaseModel):
    history: List[MessageHistoryItem]
    pagination: PaginationInfo


# --- FastAPI Application ---
app = FastAPI(
    title="Chat Service API",
    description="APIs for creating conversations, sending messages (now with streaming), and fetching chat history.",
    version="1.1.0"
)
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://localhost:3001","https://ifli.digitalfd.net:9001", "https://fg-dsm.digitalfd.net:9004", "https://fg-dsm.digitalfd.net:9001", "https://fg-dsm.digitalfd.net:9002", "https://ifli.digitalfd.net:9002"],  
    allow_origins= ["*"],
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
    expose_headers=["*"],  # Add this
    max_age=3600,# Allows all headers
)


# @app.options("/{path:path}")
# async def options_handler(path: str):
#     """Handle all OPTIONS (preflight) requests"""
#     return Response(
#         status_code=200,
#         headers={
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Max-Age": "3600",
#         }
#     )
# --- Dependency for DB session ---
def get_db():
    logging.info("Opening new database session.")
    db = SessionLocal()
    try:
        yield db
    finally:
        logging.info("Closing database session.")
        db.close()


# --- API Endpoints ---

@app.post("/conversations", response_model=CreateConversationResponse, status_code=status.HTTP_201_CREATED, tags=["Conversation"])
def create_conversation(req: CreateConversationRequest, db: Session = Depends(get_db)):
    logging.info(f"Received request to create conversation for user_id: {req.user_id}")
    
    # Check if user exists
    db_user = db.query(User).filter(User.user_id == req.user_id).first()
    if not db_user:
        logging.info(f"User with user_id: {req.user_id} not found. Creating new user.")
        db_user = User(user_id=req.user_id, customer_details=req.customer_details)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    else:
        logging.info(f"Found existing user with user_id: {req.user_id}")

    # Check if conversation already exists for this user
    existing_conversation = db.query(Conversation).filter(Conversation.user_id == req.user_id).first()
    
    if existing_conversation:
        logging.info(f"Found existing conversation {existing_conversation.conversation_id} for user {req.user_id}")
        
        # Get all messages for this conversation
        messages = db.query(Message).filter(
            Message.conversation_id == existing_conversation.conversation_id
        ).order_by(Message.created_at.asc()).all()
        
        # Get the first chatbot message as initial_message
        initial_message = next(
            (msg.message_text for msg in messages if msg.sender_id == "chatbot"),
            "Welcome back!"
        )
        
        logging.info(f"Returning existing conversation with {len(messages)} messages")
        return {
            "conversation_id": existing_conversation.conversation_id,
            "initial_message": initial_message,
            "history": messages
        }
    
    # Create new conversation if none exists
    new_conversation_id = str(uuid.uuid4())
    logging.info(f"Creating new conversation with conversation_id: {new_conversation_id}")
    db_conversation = Conversation(conversation_id=new_conversation_id, user_id=req.user_id)
    db.add(db_conversation)

    try:
        customer_name = req.customer_details.get("name", "there")
        # initial_prompt = [SystemMessage(content=f"You are a friendly customer support agent. Start the conversation with a warm welcome to {customer_name}.")]

        initial_prompt = [
            {"role": "system", "content": "You are a friendly and professional customer support agent."},
            {"role": "user", "content": f"Start the conversation with a warm, personalized welcome to {customer_name}."}
        ]
        
        logging.info("Invoking language model to generate a welcome message.")
        initial_response = llm.invoke(initial_prompt)
        welcome_message_text = getattr(initial_response, "content", None) or f"Welcome {customer_name}! How can I assist you today?"
        # welcome_message_text = initial_response.content
        logging.info("Successfully received welcome message from language model.")

        welcome_message = Message(
            message_id=str(uuid.uuid4()),
            conversation_id=new_conversation_id,
            sender_id="chatbot",
            message_text=welcome_message_text
        )
        db.add(welcome_message)
        db.commit()
        db.refresh(db_conversation)
        db.refresh(welcome_message)
        logging.info(f"Conversation {new_conversation_id} and initial message saved to database.")

    except Exception as e:
        logging.error(f"Error invoking language model or saving initial message: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Could not contact language model: {e}")

    return {
        "conversation_id": new_conversation_id,
        "initial_message": welcome_message_text,
        "history": [welcome_message]
    }


# --- NEW: Asynchronous Generator for Streaming Response ---
async def stream_chatbot_response(
    conversation_id: str,
    messages_for_llm: List[SystemMessage | AIMessage | HumanMessage],
    db: Session
) -> AsyncGenerator[str, None]:
    """
    This async generator streams the LLM response and saves the full message to the DB at the end.
    """
    full_response_text = ""
    logging.info("Starting LLM stream for conversation.")
    try:
        # Use .astream() for asynchronous streaming
        async for chunk in llm.astream(messages_for_llm):
            # The chunk object has a `content` attribute with the text
            content = chunk.content
            if content:
                full_response_text += content
                # Format the chunk as a Server-Sent Event (SSE)
                # payload = {"chatbot_response_chunk": content}
                #yield f"data: {json.dumps(payload)}\n\n"
                yield content
                # Small sleep to allow the message to be sent
                await asyncio.sleep(0.01)

    except Exception as e:
        logging.error(f"Error during LLM stream for conversation {conversation_id}: {e}", exc_info=True)
        error_payload = {"error": f"An error occurred with the language model service: {e}"}
        yield f"data: {json.dumps(error_payload)}\n\n"
        # Stop the generator and do not save to DB if an error occurs
        return

    # Once the stream is complete, save the full response to the database
    if full_response_text:
        logging.info(f"Stream complete for conversation {conversation_id}. Saving full response to database.")
        try:
            chatbot_message = Message(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                sender_id="chatbot",
                message_text=full_response_text
            )
            db.add(chatbot_message)
            db.commit()
            logging.info(f"Chatbot response for conversation {conversation_id} committed to the database.")
        except Exception as e:
            logging.error(f"Failed to save streamed response to DB for conversation {conversation_id}: {e}")
            db.rollback()
    else:
        logging.warning(f"LLM stream for conversation {conversation_id} produced an empty response.")


@app.post("/messages", tags=["Messaging"])
async def send_message(req: SendMessageRequest, db: Session = Depends(get_db)): # MODIFIED to async def
    logging.info(f"Received message from sender '{req.sender_id}' for conversation '{req.conversation_id}'")
    
    conversation = db.query(Conversation).filter(Conversation.conversation_id == req.conversation_id).first()
    if not conversation:
        logging.warning(f"Conversation not found for conversation_id: {req.conversation_id}")
        raise HTTPException(status_code=404, detail="Conversation not found")

    logging.info("Saving user message to the database.")
    user_message = Message(
        message_id=str(uuid.uuid4()),
        conversation_id=req.conversation_id,
        sender_id=req.sender_id,
        message_text=req.message_text
    )
    db.add(user_message)
    db.commit()

    logging.info(f"Fetching message history for conversation_id: {req.conversation_id} to build context.")
    history = db.query(Message).filter(Message.conversation_id == req.conversation_id).order_by(Message.created_at.desc()).limit(10).all()
    history.reverse() # Order from oldest to newest for the LLM
    logging.info(f"Found {len(history)} messages for context.")

    messages_for_llm = [SystemMessage(content="You are a helpful assistant.")]
    for msg in history:
        if msg.sender_id == 'chatbot':
            messages_for_llm.append(AIMessage(content=msg.message_text))
        else:
            messages_for_llm.append(HumanMessage(content=msg.message_text))
    
    # Create the generator and return it in a StreamingResponse
    response_generator = stream_chatbot_response(req.conversation_id, messages_for_llm, db)
    return StreamingResponse(response_generator, media_type="text/plain")


@app.get("/conversations/{conversation_id}/history", response_model=ChatHistoryResponse, tags=["Conversation"])
def get_chat_history(
    conversation_id: str,
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(50, ge=1, le=200, description="Number of messages per page"),
    db: Session = Depends(get_db)
):
    logging.info(f"Request received for chat history of conversation_id: {conversation_id} (page: {page}, page_size: {page_size})")
    
    conversation = db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
    if not conversation:
        logging.warning(f"History requested for non-existent conversation_id: {conversation_id}")
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get total count of messages
    total_messages = db.query(func.count(Message.message_id)).filter(
        Message.conversation_id == conversation_id
    ).scalar()
    
    # Calculate pagination
    total_pages = (total_messages + page_size - 1) // page_size  # Ceiling division
    offset = (page - 1) * page_size
    
    # Get paginated messages
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).offset(offset).limit(page_size).all()
    
    logging.info(f"Found {len(messages)} messages for page {page} of conversation_id: {conversation_id}. Total messages: {total_messages}")
    
    return {
        "history": messages,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_messages": total_messages,
            "total_pages": total_pages
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    logging.info("Health check endpoint was called.")
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
 
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Uvicorn server.")
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
