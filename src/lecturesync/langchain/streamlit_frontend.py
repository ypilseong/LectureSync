from rag_copy import Chatbot
from summarize_copy import DocumentSummarizer
import streamlit as st
from pydub import AudioSegment
import os
import datetime
from stt import GoogleCloudSTT

# 챗봇 인스턴스 생성

UPLOAD_DIR = "data/doc_data"
UPLOAD_TXT_DIR = "data/doc_data/summary_txt_data/"
UPLOAD_STT_DIR = "data/doc_data/stt_txt_data/"
# Streamlit 페이지 설정
st.set_page_config(page_title="LectureSync")
with st.sidebar:
    st.title('LectureSync ChatBot')

# Function for generating LLM response
def generate_response(input):
    if 'rag_bot' in st.session_state:
        result = st.session_state.rag_bot.chat(input)
        print(result)
    else:
        result = "Bot is not defined. Please upload files first."
    return result

def search_sentence(query):
    if 'rag_bot' in st.session_state:
        result, result_info = st.session_state.rag_bot.search_sentence(query)
        print(result)
    else:
        result = "Bot is not defined. Please upload files first."
    return result, result_info

def summary_doc(files):
    if 'summarizer' in st.session_state:
        result = st.session_state.summarizer.summarize(type='map_reduce')
    else:
        result = "Summarizer is not defined. Please upload files first."
    return result

def save_summary(txt):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"text_{current_time}.txt"
    file_txt_path = UPLOAD_TXT_DIR + file_name
    with open(file_txt_path, 'w', encoding='utf-8') as file:
        file.write(txt)
    return file_txt_path

def save_stt():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"text_{current_time}.txt"
    file_txt_path = UPLOAD_STT_DIR + file_name
    return file_txt_path

# Function to handle file upload and conversion
def handle_audio_video_upload(uploaded_file):
    # Save uploaded file to a specified directory
    file_extension = uploaded_file.name.split('.')[-1]
    temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    return temp_file_path

def handle_pdf_upload(uploaded_file):
    # Save uploaded PDF file to a specified directory
    temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    return temp_file_path

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요 LectureSync ChatBot 입니다. 강의 음성 파일 또는 강의 자료를 업로드 해주세요."}]
if "stt_processed_files" not in st.session_state:
    st.session_state.stt_processed_files = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []
if "video_files" not in st.session_state:
    st.session_state.video_files = []
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []
if "txt_files" not in st.session_state:
    st.session_state.txt_files = []
if "stt_files" not in st.session_state:
    st.session_state.stt_files = []
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
file_type_list = ['mp4', 'mp3', 'wav']
# File uploader for video, audio, or PDF

if not st.session_state.uploaded_files:
    uploaded_files = st.file_uploader("Upload a video, audio, or PDF file",
                                      type=["pdf", "mp4"],
                                      accept_multiple_files=True,
                                      key=st.session_state.file_uploader_key)
    st.session_state.uploaded_files = uploaded_files

    if st.session_state.uploaded_files:
        # Handle file upload and conversion
        for uploaded_file in st.session_state.uploaded_files:
            file_type = uploaded_file.type.split('/')[-1]
            
            if file_type in file_type_list:
                audio_data = handle_audio_video_upload(uploaded_file)
                st.session_state.audio_files.append(audio_data)
                st.session_state.video_files.append(audio_data)
                audio_file_name = audio_data.split('/')[-1]
                
                if audio_file_name not in st.session_state.stt_processed_files:
                    stt = GoogleCloudSTT()
                    stt_path = save_stt()
                    if audio_file_name.endswith('.mp4'):
                        audio_file_name = audio_file_name.replace('.mp4', '.wav')
                    gcs_path = f'audio-files/{audio_file_name}'
                    stt.transcribe_audio(audio_data, gcs_path, stt_path)
                    st.session_state.stt_processed_files.append(audio_file_name)
                    st.session_state.stt_files.append(stt_path)

                st.session_state.messages.append({"role": "user", "content": f"Uploaded audio/video file: {uploaded_file.name}"})
                with st.chat_message("user"):
                    st.write(f"Uploaded audio/video file: {uploaded_file.name}")

            elif file_type == 'pdf':
                pdf_data = handle_pdf_upload(uploaded_file)
                st.session_state.pdf_files.append(pdf_data)
                st.session_state.messages.append({"role": "user", "content": f"Uploaded PDF file: {uploaded_file.name}"})
                with st.chat_message("user"):
                    st.write(f"Uploaded PDF file: {uploaded_file.name}")

        with st.chat_message("assistant"):
            st.write("모든 문서 및 음성 파일이 업로드 되었습니다! 자료 요약을 시작합니다.")
        # Clear the uploaded files after processing
    

if st.session_state.uploaded_files:
    if st.button("Clear uploaded files"):
        st.session_state.file_uploader_key += 1
        st.experimental_rerun()

# Add a button to process the uploaded files
if st.session_state.pdf_files or st.session_state.stt_files:
    if st.button("자료 요약"):
        if st.session_state.pdf_files:
            st.session_state.messages.append({"role": "assistant", "content": "Processing uploaded files."})
            with st.chat_message("assistant"):
                with st.spinner("자료 요약중 ..."):
                    model_url = 'http://172.16.229.33:11436'
                    model_name = 'EEVE-Korean-Instruct-10.8B'
                    st.session_state.summarizer = DocumentSummarizer(pdf_path=st.session_state.pdf_files, stt_txt_path=st.session_state.stt_files, model_url=model_url, model_name=model_name)
                    summary = summary_doc(st.session_state.pdf_files)
                    response = f"요약이 끝났어요! 요약한 내용은 다음과 같아요: {summary}"
                    st.write(response)
                    txt_file = save_summary(response)
                    st.session_state.txt_files.append(txt_file)
            st.session_state.rag_bot = Chatbot(pdf_path=st.session_state.pdf_files, txt_path=st.session_state.txt_files, stt_txt_path=st.session_state.stt_files)
        
    
        

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant" and 'input' in locals():
    with st.chat_message("assistant"):
        with st.spinner("자료 검색을 통한 답변 진행중.."):
            if st.session_state.audio_files:
                video_url = st.session_state.video_files[0]
                st.video(video_url, format="video/mp4", start_time=0, subtitles=None, end_time=None, loop=False, autoplay=False, muted=False)
                sentence, sentence_info = search_sentence(input)
                
                if sentence_info is not None:
                    for i in range(len(sentence)):
                        with st.expander(f"문장: {i}"):
                            response = f"{sentence[i]}"
                            st.write("다음은 비디오의 해당 문장입니다:")
                            st.write(response)
                            with st.popover(f"재생 {i}"):
                                start_time = sentence_info[i]['start_time']
                                end_time = sentence_info[i]['end_time']
                                st.video(video_url, format="video/mp4", start_time=start_time, end_time=end_time, loop=False, autoplay=True, muted=False)

            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
