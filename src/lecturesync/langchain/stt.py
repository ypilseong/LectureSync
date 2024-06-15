from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
import io
import os
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# 환경 변수로 서비스 계정 키 파일 설정
google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f'gs://{bucket_name}/{destination_blob_name}'

def convert_audio_to_mono(file_path):
    """Convert audio file to mono WAV format."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # 모노로 변환
    wav_file = file_path.rsplit('.', 1)[0] + '_mono.wav'
    audio.export(wav_file, format='wav')
    return wav_file

def transcribe_audio(speech_file, output_file):
    """Transcribe the given audio file using Google Cloud Speech-to-Text API."""
    if speech_file.endswith('.mp4') or speech_file.endswith('.mp3'):
        speech_file = convert_audio_to_mono(speech_file)
    # Google Cloud Storage 버킷 이름
    bucket_name = 'lecturesync-stt'
    # 업로드할 GCS 경로
    gcs_path = 'audio-files/test_video.wav'

    # GCS에 파일 업로드
    gcs_uri = upload_to_gcs(bucket_name, speech_file, gcs_path)
    # Google Cloud 클라이언트 생성
    client = speech.SpeechClient()

    # 오디오 파일 읽기
    # with io.open(speech_file, "rb") as audio_file:
    #     content = audio_file.read()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
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
