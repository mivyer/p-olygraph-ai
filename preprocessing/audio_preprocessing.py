# preprocessing/audio_preprocessing.py

import os
import subprocess
from google.cloud import speech
from tqdm import tqdm

def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from video using FFmpeg."""
    command = [
        'ffmpeg',
        '-i', video_path,
        '-ac', '1',            # Mono channel
        '-ar', '16000',        # Sample rate
        '-vn',                 # No video
        '-y',                  # Overwrite output file
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def transcribe_audio(audio_path):
    """Transcribes audio using Google Cloud Speech-to-Text API."""
    client = speech.SpeechClient()
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_automatic_punctuation=True
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ''
    for result in response.results:
        transcript += result.alternatives[0].transcript + ' '
    return transcript.strip()

def process_videos(videos_dir, audios_dir, transcripts_dir, labels_df):
    """Processes videos: extract audio and transcribe."""
    os.makedirs(audios_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    for idx, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc='Processing videos'):
        video_id = row['video_id']
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        audio_path = os.path.join(audios_dir, f"{video_id}.wav")
        transcript_path = os.path.join(transcripts_dir, f"{video_id}.txt")
        # Extract audio
        extract_audio_from_video(video_path, audio_path)
        # Transcribe audio
        try:
            transcript = transcribe_audio(audio_path)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
        # Remove audio file if not needed
        os.remove(audio_path)
