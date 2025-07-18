# fastapi_app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
from datetime import datetime
import uuid
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import your existing components
from app.agents import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="🧳 Travel Chatbot API",
    description="A travel assistant chatbot API powered by Google Gemini & LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for chat sessions (use Redis/DB in production)
chat_sessions: Dict[str, Dict[str, Any]] = {}

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    sources_used: List[str] = []

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int
    last_activity: datetime

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str

# Helper functions
def create_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    if session_id and session_id in chat_sessions:
        chat_sessions[session_id]["last_activity"] = datetime.now()
        return session_id
    
    new_session_id = create_session_id()
    chat_sessions[new_session_id] = {
        "messages": [],
        "agent_state": {"messages": []},
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "message_count": 0
    }
    return new_session_id

def determine_sources_used(agent_state: Dict[str, Any]) -> List[str]:
    """Determine which sources were used in the response"""
    sources = []
    messages = agent_state.get("messages", [])
    
    for message in messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.name == "pdf_search":
                    sources.append("PDF Documents")
                elif tool_call.name == "web_search":
                    sources.append("Web Search")
    
    return list(set(sources))  # Remove duplicates

# API Routes
@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        session = chat_sessions[session_id]
        
        # Add user message to session
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now()
        )
        session["messages"].append(user_message.dict())
        
        # Add to agent state
        session["agent_state"]["messages"].append(HumanMessage(content=request.message))
        
        # Get response from agent
        logger.info(f"Processing message for session {session_id}: {request.message}")
        
        response = agent_app.invoke(session["agent_state"])
        
        # Extract AI response
        latest_message = response["messages"][-1]
        if hasattr(latest_message, 'content'):
            bot_response = latest_message.content
        else:
            bot_response = str(latest_message)
        
        # Update session
        session["agent_state"] = response
        session["message_count"] += 1
        session["last_activity"] = datetime.now()
        
        # Add bot message to session
        bot_message = ChatMessage(
            role="assistant",
            content=bot_response,
            timestamp=datetime.now()
        )
        session["messages"].append(bot_message.dict())
        
        # Determine sources used
        sources_used = determine_sources_used(response)
        
        logger.info(f"Response generated for session {session_id}")
        
        return ChatResponse(
            response=bot_response,
            session_id=session_id,
            timestamp=datetime.now(),
            sources_used=sources_used
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"],
        message_count=session["message_count"],
        last_activity=session["last_activity"]
    )

@app.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(session_id: str):
    """Get all messages from a session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id]["messages"]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session_data in chat_sessions.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            created_at=session_data["created_at"],
            message_count=session_data["message_count"],
            last_activity=session_data["last_activity"]
        ))
    return sessions

@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear all messages from a session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_sessions[session_id]["messages"] = []
    chat_sessions[session_id]["agent_state"] = {"messages": []}
    chat_sessions[session_id]["message_count"] = 0
    
    return {"message": f"Session {session_id} cleared successfully"}

# Background task to clean up old sessions
async def cleanup_old_sessions():
    """Clean up sessions older than 24 hours"""
    current_time = datetime.now()
    sessions_to_delete = []
    
    for session_id, session_data in chat_sessions.items():
        time_diff = current_time - session_data["last_activity"]
        if time_diff.total_seconds() > 24 * 3600:  # 24 hours
            sessions_to_delete.append(session_id)
    
    for session_id in sessions_to_delete:
        del chat_sessions[session_id]
    
    logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions")

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Travel Chatbot API started successfully!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Travel Chatbot API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )