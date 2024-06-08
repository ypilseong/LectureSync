from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama

class Chatbot:
    def __init__(self, pdf_path=None, txt_path=None, txt = None):
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self.txt = txt
        self.docs = []
        if pdf_path:
            for file_path in pdf_path:    
                pdf_loader = PyPDFLoader(file_path)
                self.docs += pdf_loader.load()
        if txt_path:
            for file_path in txt_path:
                txt_loader = TextLoader(file_path)
                self.docs += txt_loader.load()
        if txt:
            self.docs += self.txt

        self.vectorstore = None
        self.retriever = None
        if self.docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            self.splits = text_splitter.split_documents(self.docs)

            self.vectorstore = FAISS.from_documents(documents=self.splits, embedding=HuggingFaceBgeEmbeddings())
            self.retriever = self.vectorstore.as_retriever()

        self.chat_history = []

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 대답은 다음 context 기반으로 대답해. 모든 대답은 한국어(Korean)으로 대답해줘."),
            ("human", "{question}")
        ])

        self.llm = ChatOllama(base_url='http://172.16.229.33:11436',
                              model='EEVE-Korean-Instruct-10.8B')

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def add_to_history(self, question, response):
        self.chat_history.append(("human", question))
        self.chat_history.append(("assistant", response))

    def create_chain(self):
        if self.retriever:
            rag_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        return rag_chain

    def chat(self, question):
        context = "\n".join([f"{role}: {text}" for role, text in self.chat_history])
        inputs = {"context": context, "question": question}
        chain = self.create_chain()
        response = chain.invoke(inputs)
        self.add_to_history(question, response["text"])
        return response["text"]

