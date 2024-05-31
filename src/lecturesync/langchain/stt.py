import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

def convert_mp4_to_wav(mp4_path, wav_path):
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    audio.export(wav_path, format="wav")

def transcribe_audio(file_path):
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM.")
    
    model = Model("vosk-model-small-ko-0.22")
    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)
    
    result = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result.append(json.loads(recognizer.Result()))
        else:
            result.append(json.loads(recognizer.PartialResult()))
    
    result.append(json.loads(recognizer.FinalResult()))
    return " ".join([res['text'] for res in result if 'text' in res])

# 예제 사용
mp4_path = "path/to/your/lecture.mp4"
wav_path = "path/to/your/lecture.wav"

convert_mp4_to_wav(mp4_path, wav_path)
transcript = transcribe_audio(wav_path)
print(transcript)
