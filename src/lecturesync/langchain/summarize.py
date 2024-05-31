from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

from langchain_core.output_parsers import StrOutputParser


# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "./data/llm_model/"
os.environ["HF_HOME"] = "./data/llm_model/"

def summarization():

    gpu_llm = HuggingFacePipeline.from_model_id(
        model_id="beomi/Llama-3-Open-Ko-8B-Instruct-preview",  # 사용할 모델의 ID를 지정합니다.
        task="text-generation",  # 수행할 작업을 설정합니다.
        device=0,  # GPU 디바이스 번호를 지정합니다. -1은 CPU를 의미합니다.
        batch_size=2,  # 배치 크기s를 조정합니다. GPU 메모리와 모델 크기에 따라 적절히 설정합니다.
        pipeline_kwargs={
            "max_new_tokens": 512
        }  # 모델에 전달할 추가 인자를 설정합니다.
    )

    

    loader = PyPDFLoader('data/doc_data/머신러닝을활용한가짜리뷰탐지연구사용자행동분석을중심으로.pdf')
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    

    # Define prompt
    prompt_template = """Write a concise summary of the following in korean:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # 프롬프트와 언어 모델을 연결하여 체인을 생성합니다.
    gpu_chain = prompt | gpu_llm.bind(stop=["\n\n"]) | StrOutputParser()

    # Define StuffDocumentsChain
    summaries=[]
    for split in splits:
        response = gpu_chain.invoke({"text": split.page_content})
        summaries.append(response)
        print(response)
    
    final_summary = "\n\n".join(summaries)
    print(final_summary)    

    with open('data/summariztion/example.txt', 'w', encoding='utf-8') as file:
    # 변수의 내용을 파일에 쓰기
        file.write(final_summary)

    return response

summarization()
