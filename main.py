from audio_utils import AudioUtils
from speech_processor import SpeechProcessor
import torch
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"
audio_utils = AudioUtils()
speech_processor = SpeechProcessor(device)
transcriber = speech_processor.transcriber
chunk_length_s = 2.0
stream_chunk_s = 0.25

while True:
    mic = audio_utils.record_audio(transcriber, chunk_length_s, stream_chunk_s)
    print("Start speaking...")
    for item in speech_processor.transcribe_audio(mic, generate_kwargs={"max_new_tokens": 128}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break