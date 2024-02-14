from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import sounddevice

class AudioUtils:
    @staticmethod
    def record_audio(transcriber, chunk_length_s, stream_chunk_s):
        sampling_rate = transcriber.feature_extractor.sampling_rate
        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )
        return mic

    @staticmethod
    def play_audio(audio_data, samplerate):
        sf.write("temp_audio.wav", audio_data, samplerate)
        song = AudioSegment.from_wav("temp_audio.wav")
        play(song)
