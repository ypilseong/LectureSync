from google.cloud import speech_v1p1beta1 as speech
import io
import os
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# 환경 변수로 서비스 계정 키 파일 설정
google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def convert_audio(file_path):
    """Convert audio file to WAV format."""
    audio = AudioSegment.from_file(file_path)
    wav_file = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_file, format='wav')
    return wav_file

def transcribe_audio(speech_file, output_file):
    """Transcribe the given audio file using Google Cloud Speech-to-Text API."""
    if speech_file.endswith('.mp4') or speech_file.endswith('.mp3'):
        speech_file = convert_audio(speech_file)
    # Google Cloud 클라이언트 생성
    client = speech.SpeechClient()

    # 오디오 파일 읽기
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
        enable_word_time_offsets=True
    )

    # 음성 인식 요청
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=700)

    # 결과를 파일에 저장
    with open(output_file, 'w') as f:
        for result in response.results:
            alternative = result.alternatives[0]
            f.write("Transcript: {}\n".format(alternative.transcript))
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                f.write(f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}\n")


# 오디오 파일 경로
speech_file = "/home/a202121010/workspace/projects/LectureSync/data/doc_data/video_data/test_video.mp4"
output_file = '/home/a202121010/workspace/projects/LectureSync/data/doc_data/summart_txt_data/output.txt'
# 음성 인식 수행
transcribe_audio(speech_file, output_file)
