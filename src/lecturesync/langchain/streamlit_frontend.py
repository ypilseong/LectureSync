from rag import chatbot
import streamlit as st
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# 챗봇 인스턴스 생성
bot = chatbot()

# Streamlit 페이지 설정
st.set_page_config(page_title="LectureSync")
with st.sidebar:
    st.title('LectureSync ChatBot')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# Function to handle file upload and conversion
def handle_audio_video_upload(uploaded_file):
    # Save uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name
    
    # Convert to WAV if necessary
    audio = AudioSegment.from_file(temp_file_path)
    audio.export(temp_file_path + ".wav", format="wav")
    
    # Read the WAV file data
    with open(temp_file_path + ".wav", "rb") as f:
        audio_data = f.read()
    
    return audio_data

def handle_pdf_upload(uploaded_file):
    # Read PDF file data
    pdf_data = uploaded_file.getbuffer()
    return pdf_data

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요 LectureSync chatbot 입니다. 강의 음성 파일 또는 강의 자료를 업로드 해주세요."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# File uploader for video, audio, or PDF
uploaded_files = st.file_uploader("Upload a video, audio, or PDF file", type=["mp4", "mp3", "wav", "pdf"], accept_multiple_files=True)

audio_files = []
pdf_files = []

# Handle file upload and conversion
for uploaded_file in uploaded_files:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'audio' or file_type == 'video':
        audio_data = handle_audio_video_upload(uploaded_file)
        audio_files.append({"name": uploaded_file.name, "data": audio_data})
        st.session_state.messages.append({"role": "user", "content": f"Uploaded audio/video file: {uploaded_file.name}"})
        with st.chat_message("user"):
            st.write(f"Uploaded audio/video file: {uploaded_file.name}")
    
    elif uploaded_file.type == 'application/pdf':
        pdf_data = handle_pdf_upload(uploaded_file)
        pdf_files.append({"name": uploaded_file.name, "data": pdf_data})
        st.session_state.messages.append({"role": "user", "content": f"Uploaded PDF file: {uploaded_file.name}"})
        with st.chat_message("user"):
            st.write(f"Uploaded PDF file: {uploaded_file.name}")

# Process the uploaded files (stub for actual processing)
if audio_files or pdf_files:
    st.session_state.messages.append({"role": "assistant", "content": "Processing uploaded files."})
    with st.chat_message("assistant"):
        with st.spinner("Processing your files..."):
            # Here you can add the actual processing logic
            response = f"Files processed successfully. Audio files: {len(audio_files)}, PDF files: {len(pdf_files)}."
            st.write(response)

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
