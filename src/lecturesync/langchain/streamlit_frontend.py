from rag_copy import Chatbot
from summarize_copy import DocumentSummarizer
import streamlit as st
from pydub import AudioSegment
import os
from io import BytesIO
import chardet
import datetime

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
        result = st.session_state.rag_bot.search_sentence(query)
        print(result)
    else:
        result = "Bot is not defined. Please upload files first."
    return result

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
    
    # Convert to WAV if necessary
    audio = AudioSegment.from_file(temp_file_path)
    wav_file_path = temp_file_path + ".wav"
    audio.export(wav_file_path, format="wav")
    
    return wav_file_path

def handle_pdf_upload(uploaded_file):
    # Save uploaded PDF file to a specified directory
    temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    return temp_file_path

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요 LectureSync ChatBot 입니다. 강의 음성 파일 또는 강의 자료를 업로드 해주세요."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# File uploader for video, audio, or PDF
uploaded_files = st.file_uploader("Upload a video, audio, or PDF file", type=[ "pdf"], accept_multiple_files=True)

file_type_list = ["mp4", "mp3", "wav"]
audio_files = []
pdf_files = []
txt_files = []
if uploaded_files:
    # Handle file upload and conversion
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type.split('/')[-1]
        
        # if file_type in file_type_list:
        #     audio_data = handle_audio_video_upload(uploaded_file)
        #     audio_files.append(audio_data)
        #     st.session_state.messages.append({"role": "user", "content": f"Uploaded audio/video file: {uploaded_file.name}"})
        #     with st.chat_message("user"):
        #         st.write(f"Uploaded audio/video file: {uploaded_file.name}")
        
        # elif file_type == 'pdf':
        #     pdf_data = handle_pdf_upload(uploaded_file)
        #     pdf_files.append(pdf_data)
        #     st.session_state.messages.append({"role": "user", "content": f"Uploaded PDF file: {uploaded_file.name}"})
        #     with st.chat_message("user"):
        #         st.write(f"Uploaded PDF file: {uploaded_file.name}")

        pdf_data = handle_pdf_upload(uploaded_file)
        pdf_files.append(pdf_data)
        st.session_state.messages.append({"role": "user", "content": f"Uploaded PDF file: {uploaded_file.name}"})
        with st.chat_message("user"):
            st.write(f"Uploaded PDF file: {uploaded_file.name}")
            
    # Add a button to process the uploaded files
    if st.button("Process Files"):
        if pdf_files:
            st.session_state.messages.append({"role": "assistant", "content": "Processing uploaded files."})
            with st.chat_message("assistant"):
                with st.spinner("Processing your files..."):
                    model_url = 'http://172.16.229.33:11436'
                    model_name = 'EEVE-Korean-Instruct-10.8B'
                    st.session_state.summarizer = DocumentSummarizer(pdf_path=pdf_files, model_url=model_url, model_name=model_name)
                    summary = summary_doc(pdf_files)
                    response = f"요약이 끝났어요! 요약한 내용은 다음과 같아요: {summary}"
                    st.write(response)
                    txt_file = save_summary(response)
                    txt_files.append(txt_file)
                    stt_txt_path = ['data/doc_data/stt_txt_data/sentences.txt']
            st.session_state.rag_bot = Chatbot(pdf_path=pdf_files, txt_path=txt_files, stt_txt_path = stt_txt_path)
        
        # Clear the uploaded files
        st.session_state.uploaded_files = []
else:
    st.session_state.rag_bot = Chatbot()

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant" and 'input' in locals():
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

# 문장 검색을 트리거하는 버튼
query = st.text_input("검색할 문장을 입력하세요:")
if st.button("문장 검색"):
    if query:
        with st.spinner("문장을 검색하는 중..."):
            search_results = search_sentence(query)
            st.write(search_results)