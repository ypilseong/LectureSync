from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
import re
import torch


class Chatbot:
    def __init__(self, pdf_path=None, txt_path=None, stt_txt_path=None, chunk_size=2000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self.stt_txt_path = stt_txt_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs_pdf = []
        self.docs_stt = []
        self.docs = []

        if pdf_path:
            for file_path in self.pdf_path:
                pdf_loader = PyPDFLoader(file_path)
                self.docs_pdf += pdf_loader.load()
        
        if txt_path:
            for file_path in self.txt_path:
                txt_loader = TextLoader(file_path)
                self.docs += txt_loader.load()
        
        if stt_txt_path:
            for file_path in self.stt_txt_path:
                txt_loader = TextLoader(file_path)
                self.docs_stt += txt_loader.load()
        
        self.vectorstore = None
        self.retriever = None
        
        all_docs = []
        if self.docs_pdf:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\n\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            self.splits_pdf = text_splitter.split_documents(self.docs_pdf)
            all_docs.extend(self.splits_pdf)
        
        if self.docs:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\n\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            self.splits_doc = text_splitter.split_documents(self.docs)
            all_docs.extend(self.splits_doc)
        if self.docs_stt:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\n\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            self.splits_stt = text_splitter.split_documents(self.docs_stt)
            all_docs.extend(self.splits_stt)

        if all_docs:
            self.vectorstore = FAISS.from_documents(documents=all_docs, embedding=HuggingFaceBgeEmbeddings())
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.chat_history = []

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자.
             대답은 다음 '{context}' 기반으로 대답해. 모든 대답은 한국어(Korean)으로 대답해줘.
             사용자가 질문하고 너가 답한 내용은 '{chat_history}' 이러한 내용이야.
             만약 답변할 정보가 없거나 확실하지 않으면 '정보를 찾을 수 없습니다' 또는 '확실하지 않습니다'라고 대답해."""),
            ("human", "{question}")
        ])

        # self.llm = ChatOllama(base_url='http://172.16.229.33:11436',
        #                       model='EEVE-Korean-Instruct-10.8B',
        #                       temperature=0.4)
        self.llm = ChatOpenAI(temperature=0.1,
                              model_name='gpt-4o')

        # STT 관련 변수 초기화
        self.sentences_data = self.parse_transcription_file(self.stt_txt_path[0] if self.stt_txt_path else None)
        if self.sentences_data:
            self.corpus = [sentence["text"] for sentence in self.sentences_data]
            self.corpus_embeddings = HuggingFaceBgeEmbeddings().embed_documents(self.corpus)
        else:
            self.corpus = []
            self.corpus_embeddings = None

    @staticmethod
    def parse_transcription_file(file_path):
        if not file_path:
            return None
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        sentence_pattern = re.compile(r"Transcript: (.*?)\nStart time: ([\d.]+), End time: ([\d.]+)\n", re.DOTALL)
        sentences_info = sentence_pattern.findall(content)

        sentences_data = []
        for sentence, start_time, end_time in sentences_info:
            sentences_data.append({
                "text": sentence.strip(),
                "start_time": float(start_time),
                "end_time": float(end_time)
            })

        return sentences_data

    def find_sentence_time(self, query):
        if not self.corpus_embeddings:
            return "문장 검색이 불가능합니다.", None
        query_embedding = HuggingFaceBgeEmbeddings().embed_documents([query])
        query_embedding = torch.tensor(query_embedding)
        hits = util.semantic_search(query_embedding, torch.tensor(self.corpus_embeddings), top_k=3)
        hits = hits[0]

        response = []
        response_info = []
        for hit in hits:
            sentence_info = self.sentences_data[hit['corpus_id']]
            response.append(f"Sentence: {sentence_info['text']}, Start Time: {sentence_info['start_time']}, End Time: {sentence_info['end_time']}")
            response_info.append(sentence_info)
        return response, response_info

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def add_to_history(self, question, response):
        self.chat_history.append(("human", question))
        self.chat_history.append(("assistant", response))

    def create_chain(self):
        if self.retriever:
            rag_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        return rag_chain

    def chat(self, question):
        # chat_history = ''
        # if self.chat_history is not None:
        chat_history = "\n".join([f"{role}: {text}" for role, text in self.chat_history])
        inputs = {"question": question, "chat_history": chat_history}
        chain = self.create_chain()
        response = chain.invoke(question)
        self.add_to_history(question, response)
        return response

    def search_sentence(self, question):
        
        sentence, time_info = self.find_sentence_time(question)
        
        return sentence, time_info