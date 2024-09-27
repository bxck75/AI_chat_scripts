
import asyncio
import aiohttp
import pyaudio
import numpy as np
import wave
import tempfile
import os

from gradio_client import Client, handle_file
from scipy.io.wavfile import write

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/starchat2-15b-v0.1/v1/chat/completions"
API_TOKEN = key  # Replace with your actual token
# filename: whisper_transcribe.py
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
from rich import print as rp



def play_audio(filename):
    # Open the wav file
    wf = wave.open(filename, 'rb')

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    # Open a stream with the appropriate format
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read data from the wav file in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    # Play the audio stream
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio
    p.terminate()
    

############################# TXT To Speech #########################################
# list all voice repos
client = Client("k2-fsa/text-to-speech")
result = client.predict(
		language="English",
		api_name="/update_model_dropdown")

voice = result.get('choices')[1][0] # speaker 1 from third repo\

input_text="""
    From an evolutionary perspective, birds evolved from reptiles, and eggs existed long before chickens. 
    The first "chicken" would have been born from an egg laid by an ancestor that was not quite a chicken but close enough genetically to produce one.
""",
audio_result_path, html = client.predict(
		language="English",
		repo_id=voice,"csukuangfj/vits-piper-en_US-glados|1 speaker",
		text=input_text,
		sid="1",
		speed=0.9,
		api_name="/process"
)
rp(audio_result_path) 

play_audio(audio_result_path)

#################### Speech To TXT  ############################################
""" 
api_name: /process_uploaded_file
"""

repos=['tiny.en', 'base.en', 'small.en', 'medium.en', 'distil-medium.en', 'tiny', 'base', 'small', 'distil-small.en', 'medium', 'medium-aishell']

client = Client("k2-fsa/automatic-speech-recognition-with-whisper")
text,html = client.predict(
		repo_id="tiny.en",
		in_filename=handle_file(audio_result_path),
		api_name="/process_uploaded_file"
)
rp(text)
""" Accepts 2 parameters:
repo_id Literal
    ['tiny.en', 'base.en', 'small.en', 'medium.en', 'distil-medium.en', 'tiny', 'base', 'small', 'distil-small.en', 'medium', 'medium-aishell'] \   
    Default: "tiny.en"

in_filename 
    filepath 
        Required

Returns tuple of 2 elements
[0] str The output value Textbox component.
[1] str The output value Html component.


api_name: /process_microphone"""
client = Client("k2-fsa/automatic-speech-recognition-with-whisper")
result = client.predict(
		repo_id="tiny.en",
		in_filename=handle_file('/mnt/04ef09de-2d9f-4fc2-8b89-de7dc0155e26/new_code/NewChat/VoiceChat/voice_files/miep21.wav'),
		api_name="/process_microphone"
)
print(result)
""" Accepts 2 parameters:
repo_id Literal
    ['tiny.en', 'base.en', 'small.en', 'medium.en', 'distil-medium.en', 'tiny', 'base', 'small', 'distil-small.en', 'medium', 'medium-aishell'] 
    Default: "tiny.en"

in_filename 
    filepath 
        Required

Returns tuple of 2 elements
[0] str The output value Textbox component.
[1] str The output value Html component. """