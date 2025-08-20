import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

# Import the new, refactored functions from your chatbot script
from sts_chatbot import process_query, generate_audio_for_text

app = FastAPI(
    title="Medical Chatbot API",
    description="API for processing text/audio and generating speech.",
    version="2.2.0" 
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response Validation ---

class TextPayload(BaseModel):
    """Request model for endpoints that accept text."""
    text: str

class QueryPayload(BaseModel):
    """Request model for the text processing endpoint."""
    query: str

class TextResponse(BaseModel):
    """Response model for endpoints that return a text answer."""
    answer: str

class AudioResponse(BaseModel):
    """Response model for the audio generation endpoint."""
    audio_data: str # This will be a Base64 encoded string

# --- API Endpoints ---

@app.get("/")
def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Medical Chatbot API is running"}

@app.post("/chat-text", response_model=TextResponse)
async def handle_text_query(payload: QueryPayload):
    """
    Endpoint for processing a text-based query.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    answer = process_query(payload.query, input_type="text")

    if not answer:
        raise HTTPException(status_code=500, detail="Failed to get a response from the chatbot.")

    return {"answer": answer}


@app.post("/chat-audio", response_model=TextResponse)
async def handle_audio_query(file: UploadFile = File(...)):
    """
    Endpoint for processing an audio file query in memory.
    """
    try:
        # Read file content into an in-memory bytes buffer
        audio_bytes = await file.read()
        
        # Pass the bytes to the processing function
        # NOTE: You will need to update `process_query` to accept bytes instead of a file path for audio.
        # For now, I'm assuming it can handle a file-like object.
        audio_file_like_object = io.BytesIO(audio_bytes)
        
        # If your process_query function absolutely needs a file path, 
        # you might need to save it temporarily, but this is not ideal on Vercel.
        # The best approach is to adapt process_query.
        
        # This is a placeholder for how you might adapt it.
        # You'll need to see what `process_query` expects.
        # Let's assume for now it can take a file-like object.
        answer = process_query(audio_file_like_object, input_type="audio")

        if not answer:
            raise HTTPException(status_code=500, detail="Failed to process the audio file.")

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/generate-audio", response_model=AudioResponse)
async def handle_audio_generation(payload: TextPayload):
    """
    Endpoint for generating audio from text.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text for audio generation cannot be empty.")

    try:
        response_dict = generate_audio_for_text(payload.text)

        if response_dict and isinstance(response_dict.get("audio_data"), str) and response_dict["audio_data"]:
            return {"audio_data": response_dict["audio_data"]}
        else:
            raise HTTPException(status_code=500, detail="Audio generation returned invalid or empty data.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred during audio generation: {str(e)}")
