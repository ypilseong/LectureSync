from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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
    vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFaceBgeEmbeddings())

    # 단계 4: 검색(Search)
    retriever = vectorstore.as_retriever()
    
    # 단계 5: 프롬프트 생성(Create Prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 대답은 다음 context 기반으로 대답해. 모든 대답은 한국어(Korean)으로 대답해줘. Context: {context} "),
        ("human", "{question}")
    ])
    
    

    # 단계 6: 언어모델 생성(Create LLM)
    tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B-Instruct-preview",
                                              padding_side='left',
                                            )
    model = AutoModelForCausalLM.from_pretrained("beomi/Llama-3-Open-Ko-8B-Instruct-preview")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_length=2048,truncation=True,device=0)
    gpu_llm = HuggingFacePipeline(
        pipeline = pipe,
        batch_size=2,  # 배치 크기s를 조정합니다. GPU 메모리와 모델 크기에 따라 적절히 설정합니다.
        pipeline_kwargs={
            "max_new_tokens": 512,
            "max_length": 1024,
            "do_sample": True,
            "temperature": 0.5
        }  # 모델에 전달할 추가 인자를 설정합니다.
    )


    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)


    # 단계 7: 체인 생성(Create Chain)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | gpu_llm
        | StrOutputParser()
    )
