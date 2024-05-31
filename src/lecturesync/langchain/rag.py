from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class chatbot():

    pdf_path = ''
    txt_path = ''

    # 단계 1: 문서 불러오기(docs load)
    pdf_loader = PyPDFLoader(pdf_path)
    txt_loader = TextLoader(txt_path)

    pdf_docs = pdf_loader.load()
    txt_docs = txt_loader.load()

    docs = pdf_docs + txt_docs


    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    splits = text_splitter.split_documents(docs)

    # 단계 3: 임베딩 & 벡터스토어 생성(Create Vectorstore)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # 단계 4: 검색(Search)
    retriever = vectorstore.as_retriever()

    # 단계 5: 프롬프트 생성(Create Prompt)
    prompt = hub.pull("rlm/rag-prompt")

    # 단계 6: 언어모델 생성(Create LLM)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)


    # 단계 7: 체인 생성(Create Chain)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
