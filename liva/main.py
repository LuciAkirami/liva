from audio_utils import AudioUtils
from speech_processor import SpeechProcessor
from llm_inference import LLMInference
from speech_generator import SpeechGenerator
import argparse
import torch
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        help="Any OpenAI Compatible Chat Endpoint. For example: Ollama - http://localhost:8000/v1, vllm endpoints",
        default='http://localhost:11434/v1'
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="Any LLM model to power the Text Generator. Use this option when workig with Ollama",
        default='mistral:instruct'
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Provide the API Key for the relevant Chat Endpoint URL",
        default='ollama'
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        help="Provide the Speech2Text model. For example openai/whisper-base.en",
        default="openai/whisper-base.en"
    )

    args = parser.parse_args()
    # print(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    audio_utils = AudioUtils()
    speech_processor = SpeechProcessor(
        device = device,
        stt_model = args.stt_model
        )
    model = LLMInference(
        model_id = args.model_id,
        url = args.url,
        api_key = args.api_key
        )
    audio_model = SpeechGenerator(device)

    transcriber = speech_processor.transcriber
    chunk_length_s = 2.0
    stream_chunk_s = 0.25

    while True:
        mic = audio_utils.record_audio(transcriber, chunk_length_s, stream_chunk_s)
        # print("Start speaking...")
        print("User:")

        for item in speech_processor.transcribe_audio(mic, generate_kwargs={"max_new_tokens": 128}):
            sys.stdout.write("\033[K")
            print(item["text"], end="\r")
            if not item["partial"][0]:
                break
    
        print(item['text'].strip(),end='\n\n')

        response = model.query_model(item['text'])

        print("LIVA:")
        print(response.strip(),end='\n\n')

        speech = audio_model.synthesize_audio(response)
        audio_utils.play_audio(speech['audio'],speech['sampling_rate'])
    
if __name__ == "__main__":
    main()