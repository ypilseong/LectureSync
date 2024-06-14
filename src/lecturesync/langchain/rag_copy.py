from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama

class Chatbot:
    def __init__(self, pdf_path=None, txt_path=None, txt=None, chunk_size=2000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self.txt = txt
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs_pdf = []
        self.docs = []
        if pdf_path:
            for file_path in self.pdf_path:
                pdf_loader = PyPDFLoader(file_path)
                self.docs_pdf += pdf_loader.load()
        
        if txt_path:
            for file_path in self.txt_path:
                txt_loader = TextLoader(file_path)
                self.docs += txt_loader.load()
        
        
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
        
        if all_docs:
            self.vectorstore = FAISS.from_documents(documents=all_docs, embedding=HuggingFaceBgeEmbeddings())
            self.retriever = self.vectorstore.as_retriever()

        self.chat_history = []

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 대답은 다음 '{context}' 기반으로 대답해. 모든 대답은 한국어(Korean)으로 대답해줘. 사용자가 질문하고 너가 답한 내용은 '{chat_history}' 이러한 내용이야."),
            ("human", "{question}")
        ])

        self.llm = ChatOllama(base_url='http://172.16.229.33:11436',
                              model='EEVE-Korean-Instruct-10.8B',
                              temperature=0.4)

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
