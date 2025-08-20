# groq_stt.py

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the Groq client with your API key from the environment variables
# Note: The api_key parameter is not needed if the GROQ_API_KEY environment variable is set.
# However, it's good practice to include it for clarity.
try:
    groq_client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )
except Exception as e:
    print(f"❌ Error initializing Groq client: {e}")
    groq_client = None


def transcribe_audio_with_groq(audio_path: str) -> str:
    """
    Transcribes an audio file to text using the Groq API.

    Args:
        audio_path: The file path to the audio file to be transcribed.
                    Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

    Returns:
        The transcribed text as a string, or an empty string if transcription fails.
    """
    if groq_client is None:
        print("❌ Groq client not initialized. Cannot transcribe audio.")
        return ""

    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at {audio_path}")
        return ""

    print("🎙️ Sending audio for transcription to Groq...")
    
    try:
        # Use 'with open' to ensure the file is properly closed
        with open(audio_path, "rb") as audio_file:
            # Call the Groq transcription API
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3", # Use a whisper model for transcription
                response_format="text", # Request a plain text response
            )
        
        # The transcription object has a `text` attribute with the transcribed content
        return transcription.text

    except Exception as e:
        print(f"❌ Error during Groq transcription: {e}")
        return ""


# Example of how to use the function locally
if __name__ == "__main__":
    # Create a dummy audio file for testing purposes
    # In a real app, you would have a recorded file here
    print("This is a mock run to test the function.")
    print("Please ensure you have an audio file and replace 'your_audio.mp3' for a real test.")
    
    # You would replace 'your_audio.mp3' with a real path to an audio file
    mock_audio_path = "C:\Desktop\MedicoBotBackend\demo_speaker0.mp3" 
    
    # This part of the code is for demonstration only and will not work without a valid audio file
    if os.path.exists(mock_audio_path):
        transcribed_text = transcribe_audio_with_groq(mock_audio_path)
        if transcribed_text:
            print(f"✅ Transcription successful: {transcribed_text}")
        else:
            print("❌ Transcription failed.")
    else:
        print(f"Test file '{mock_audio_path}' not found. Skipping test.")

