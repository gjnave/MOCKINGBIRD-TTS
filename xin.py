import gradio as gr
import requests
import tempfile
import os
from pydub import AudioSegment
import random
import yt_dlp
import pynvml
import torch
import gc

# Define the TTS server URL
url = "http://localhost:8020/tts_to_audio/"

# Define the directory containing the speaker .wav files
speaker_directory = "xtts-api-server\\speakers"

# Get the list of .wav files in the directory
def get_speaker_files():
    return [f for f in os.listdir(speaker_directory) if f.endswith('.wav')]

# Initialize the speaker_wav variable
speaker_wav = get_speaker_files()[0]
language = "en"

# Initialize NVML
pynvml.nvmlInit()

def get_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming we're using the first GPU
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / info.total  # Return memory usage as a fraction

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def get_random_video_segment(video_url, segment_duration_ms=8000):
    # Download video
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': 'temp_audio.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    # Load audio
    audio = AudioSegment.from_wav("temp_audio.wav")
    
    # Cut random segment
    audio_duration_ms = len(audio)
    if audio_duration_ms <= segment_duration_ms:
        raise ValueError("The audio file is shorter than the segment duration.")
    start_time_ms = random.randint(0, audio_duration_ms - segment_duration_ms)
    segment = audio[start_time_ms:start_time_ms + segment_duration_ms]
    
    # Export segment
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    segment.export(output_path, format="wav")
    
    return output_path



# Function to cut a random 8-second segment
def cut_random_segment(input_audio, output_audio_path, segment_duration_ms=8000):
    audio = AudioSegment.from_file(input_audio)
    audio_duration_ms = len(audio)
    if audio_duration_ms <= segment_duration_ms:
        raise ValueError("The audio file is shorter than the segment duration.")
    start_time_ms = random.randint(0, audio_duration_ms - segment_duration_ms)
    segment = audio[start_time_ms:start_time_ms + segment_duration_ms]
    segment.export(output_audio_path, format="wav")

# Function to send text to TTS server and get audio
def text_to_speech(text, selected_speaker, uploaded_speaker, video_url):
    global speaker_wav
    
    # Check GPU memory usage
    if get_gpu_memory_usage() > 0.9:  # If more than 90% of VRAM is used
        clear_gpu_memory()
    
    if video_url:
        speaker_wav = get_random_video_segment(video_url)
    elif uploaded_speaker:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_speaker)
            temp_file_path = temp_file.name
        # Cut a random segment from the uploaded file
        segment_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        cut_random_segment(temp_file_path, segment_path)
        speaker_wav = segment_path
    else:
        speaker_wav = selected_speaker

    payload = {
        "text": text,
        "speaker_wav": speaker_wav,
        "language": language
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            # Save the audio content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            return temp_file_path
        else:
            return f"Error: {response.status_code}, {response.content}"
    except Exception as e:
        return f"Request failed: {str(e)}"
    finally:
        # Check GPU memory usage again after processing
        if get_gpu_memory_usage() > 0.9:
            clear_gpu_memory()

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Cognibuild.ai Quick & Easy TTS ")
    gr.Markdown("Enter text and hear it spoken aloud. Select a speaker from the dropdown, upload a custom speaker .wav file, or provide a video link.")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter Text")
        speaker_dropdown = gr.Dropdown(label="Select Speaker", choices=get_speaker_files(), value=speaker_wav)
    
    with gr.Row():
        uploaded_speaker = gr.File(label="Upload Custom Speaker .wav", type="binary")
        video_url = gr.Textbox(label="Video URL")
    
    generate_btn = gr.Button("Generate Audio")
    clear_vram_btn = gr.Button("Clear VRAM")
    audio_output = gr.Audio(label="Generated Audio")
    
    def manual_clear_vram():
        clear_gpu_memory()
        return "VRAM cleared"
    
    generate_btn.click(text_to_speech, inputs=[text_input, speaker_dropdown, uploaded_speaker, video_url], outputs=audio_output)
    clear_vram_btn.click(manual_clear_vram, inputs=[], outputs=gr.Textbox(label="VRAM Status"))

iface.launch(open_browser=True)
