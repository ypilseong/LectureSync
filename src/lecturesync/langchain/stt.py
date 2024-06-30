from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from pydub import AudioSegment
from dotenv import load_dotenv
import os

load_dotenv()

class GoogleCloudSTT:
    def __init__(self, credentials_path=None):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        self.storage_client = storage.Client()
        self.speech_client = speech.SpeechClient()

    def upload_to_gcs(self, bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to Google Cloud Storage."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        return f'gs://{bucket_name}/{destination_blob_name}'

    def convert_audio_to_mono(self, file_path):
        """Convert audio file to mono WAV format."""
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)  # 모노로 변환
        wav_file = file_path.rsplit('.', 1)[0] + '_mono.wav'
        audio.export(wav_file, format='wav')
        return wav_file

    def transcribe_audio(self, speech_file, gcs_path,  output_file_sentences):
        """Transcribe the given audio file using Google Cloud Speech-to-Text API."""
        if speech_file.endswith('.mp4') or speech_file.endswith('.mp3') or speech_file.endswith('.wav'):
            speech_file = self.convert_audio_to_mono(speech_file)
        
        # GCS에 파일 업로드
        gcs_uri = self.upload_to_gcs('lecturesync-stt', speech_file, gcs_path)
        
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ko-KR",
            enable_word_time_offsets=True
        )

        # 음성 인식 요청
        operation = self.speech_client.long_running_recognize(config=config, audio=audio)

        print("Waiting for operation to complete...")
        response = operation.result(timeout=700)

        # 문장 단위로 시작과 끝 시간 저장
        with open(output_file_sentences, 'w') as f:
            for result in response.results:
                alternative = result.alternatives[0]
                start_time = alternative.words[0].start_time.total_seconds()
                end_time = alternative.words[-1].end_time.total_seconds()
                f.write(f"Transcript: {alternative.transcript}\n")
                f.write(f"Start time: {start_time}, End time: {end_time}\n")

# 사용 예시
if __name__ == "__main__":
    stt = GoogleCloudSTT()
    speech_file = "/home/a202121010/workspace/projects/LectureSync/data/doc_data/video_data/test_video.mp4"
    output_file = '/home/a202121010/workspace/projects/LectureSync/data/doc_data/summary_txt_data/output2.txt'
    output_file_sentences = '/home/a202121010/workspace/projects/LectureSync/data/doc_data/stt_txt_data/sentences.txt'
    gcs_path = 'audio-files/test_video.wav'
    
    stt.transcribe_audio(speech_file, gcs_path, output_file, output_file_sentences)
