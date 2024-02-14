from transformers import pipeline

class SpeechProcessor:
    def __init__(self, device):
        self.classifier = pipeline(
            "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
        )
        self.transcriber = pipeline(
            "automatic-speech-recognition", model="openai/whisper-base.en", device=device
        )

    def classify_audio(self, audio_data):
        return self.classifier(audio_data)

    def transcribe_audio(self, audio_data,generate_kwargs):
        return self.transcriber(audio_data,generate_kwargs=generate_kwargs)