# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate

# from whisper_med_stt import transcribe_audio
# from tts_utils import generate_audio_from_api

# # -------------------- Load ENV --------------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# INDEX_NAME = "medical-chatbot"

# # -------------------- Init Pinecone v4 SDK --------------------
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# # -------------------- Embeddings --------------------
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # -------------------- LLM Setup --------------------
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash-latest",
#     api_key=GEMINI_API_KEY,
#     temperature=0.0
# )

# # -------------------- Prompt Template --------------------
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a professional medical AI assistant. Your task is to answer user medical questions strictly based on the given CONTEXT below.
# If the answer cannot be found in the CONTEXT, reply clearly with "I don't know".
# Be precise, factual, and clear. Avoid assumptions. Do not generate medical advice unless it is explicitly in the CONTEXT.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# ANSWER:
# """
# )

# # -------------------- RAG Chain (Manual Pinecone query) --------------------
# def get_answer_from_gemini(user_question: str):
#     if not user_question.strip():
#         return "Please ask a valid medical question."

#     # Get vector
#     question_vector = embedding_model.embed_query(user_question)

#     # Query Pinecone (manual)
#     search_result = index.query(
#         vector=question_vector,
#         top_k=5,
#         include_metadata=True
#     )

#     # Extract context
#     docs = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
#     context = "\n\n".join(docs) if docs else "No relevant documents found."

#     # Prepare prompt
#     final_prompt = prompt_template.format(context=context, question=user_question)

#     # Get answer from Gemini
#     response = llm.invoke(final_prompt)
#     return str(response.content)

# # 🔁 WRAPPER FUNCTION FOR FASTAPI
# def get_speech_to_speech_response(input_data, input_type="text"):
#     """
#     Processes a query and returns a response.
#     - For "text", it returns the text answer string.
#     - For "audio", it returns a dictionary with the text answer and Base64 audio data.
#     """
#     if input_type == "audio":
#         user_question = transcribe_audio(input_data)
#     else:
#         user_question = input_data

#     if not user_question or not user_question.strip():
#         error_text = "I'm sorry, I couldn't understand the input. Please try again."
#         audio_data = generate_audio_from_api(error_text) if input_type == "audio" else None
#         return {"text_answer": error_text, "audio_data": audio_data} if input_type == "audio" else None

#     answer = get_answer_from_gemini(user_question)

#     if input_type == "audio":
#         output_audio = generate_audio_from_api(answer)
#         return {"text_answer": answer, "audio_data": output_audio}
#     else:
#         return answer



#######################################################################################

import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Assuming these are your custom utility modules
from groq_stt import transcribe_audio_with_groq
from tts_utils import generate_audio_from_api

# -------------------- Load ENV --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "medical-chatbot"

# -------------------- Init Pinecone v4 SDK --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------- Embeddings --------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- LLM Setup --------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    api_key=GEMINI_API_KEY,
    temperature=0.0
)

# -------------------- Prompt Template --------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a professional medical AI assistant. Your task is to answer user medical questions strictly based on the given CONTEXT below.
If the answer cannot be found in the CONTEXT, reply clearly with "I don't know".
Be precise, factual, and clear. Avoid assumptions. Do not generate medical advice unless it is explicitly in the CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
)

# -------------------- RAG Chain (Manual Pinecone query) --------------------
def get_answer_from_gemini(user_question: str):
    if not user_question.strip():
        return "Please ask a valid medical question."

    # Get vector
    question_vector = embedding_model.embed_query(user_question)

    # Query Pinecone (manual)
    search_result = index.query(
        vector=question_vector,
        top_k=5,
        include_metadata=True
    )

    # Extract context
    docs = [match['metadata'].get('text', '') for match in search_result.get('matches', [])]
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    # Prepare prompt
    final_prompt = prompt_template.format(context=context, question=user_question)

    # Get answer from Gemini
    response = llm.invoke(final_prompt)
    return str(response.content)

# -------------------- MODIFIED AND NEW WRAPPER FUNCTIONS --------------------

# 🔁 MODIFIED: This function now only processes the query and returns text.
def process_query(input_data, input_type="text"):
    """
    Processes a text or audio query and returns ONLY the text-based answer.
    - For "text", it uses the input string directly.
    - For "audio", it first transcribes the audio to text using Groq.
    """
    user_question = ""
    if input_type == "audio":
        # Transcribe the audio file/data to get the user's question
        # Call the new function here!
        user_question = transcribe_audio_with_groq(input_data)
    else:
        # Use the text input directly
        user_question = input_data

    # Handle cases where transcription fails or input is empty
    if not user_question or not user_question.strip():
        return "I'm sorry, I couldn't understand the input. Please try again or ask a valid question."

    # Get the text answer from the RAG pipeline
    answer = get_answer_from_gemini(user_question)
    return answer

# ✨ NEW: This function is dedicated to generating audio from a given text.
def generate_audio_for_text(text_to_speak: str):
    """
    Takes a string of text and converts it to audio data.
    This should be called separately when the user wants to hear the response.
    """
    if not text_to_speak or not text_to_speak.strip():
        return None  # Or handle as an error

    # Call your TTS utility to get the audio data
    output_audio = generate_audio_from_api(text_to_speak)

    # Return the audio data, likely in a JSON-friendly format for your API
    return {"audio_data": output_audio}


