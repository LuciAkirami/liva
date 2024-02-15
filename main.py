from audio_utils import AudioUtils
from speech_processor import SpeechProcessor
from llm_inference import LLMInference
from speech_generator import SpeechGenerator
import torch
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"

audio_utils = AudioUtils()
speech_processor = SpeechProcessor(device)
model = LLMInference()
audio_model = SpeechGenerator(device)

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
    
    response = model.query_model(item['text'])
    print(response)
    speech = audio_model.synthesize_audio(response)
    audio_utils.play_audio(speech['audio'],speech['sampling_rate'])
    