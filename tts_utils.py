import os
import base64
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()
elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

def generate_audio_from_api(text: str):
    print("🔊 Generating audio for text...")
    try:
        audio_bytes = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id="i4CzbCVWoqvD0P1QJCUL",  
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        return play(audio_bytes)

    except Exception as e:
        print(f"❌ Error generating audio from ElevenLabs: {e}")
        return None