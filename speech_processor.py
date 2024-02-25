from transformers import pipeline

class SpeechProcessor:
    def __init__(self, device, stt_model = "openai/whisper-base.en"):
        self.stt_model = stt_model
        self.classifier = pipeline(
            "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=device
        )
        self.transcriber = pipeline(
            "automatic-speech-recognition", model=self.stt_model, device=device
        )

    def classify_audio(self, audio_data):
        return self.classifier(audio_data)

    def transcribe_audio(self, audio_data,generate_kwargs):
        return self.transcriber(audio_data,generate_kwargs=generate_kwargs)