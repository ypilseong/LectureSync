from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain
import os
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.chat_models import ChatOpenAI
# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "./data/llm_model/alphako/"
os.environ["HF_HOME"] = "./data/llm_model/alphako/"

def summarization():
    # ollama api
    llm = ChatOllama(base_url='http://172.16.229.33:11436',
                        model='EEVE-Korean-Instruct-10.8B',
                        temperature=0
                        )
    # llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125' )
    loader = PyPDFLoader('data/doc_data/test_doc.pdf')
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,   # 사이즈
        chunk_overlap=200, # 중첩 사이즈
        is_separator_regex=False 
    )
    splits = text_splitter.split_documents(docs)
    
    print(f'총 분할된 도큐먼트 수: {len(splits)}')

    document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
    )
    
    # Define prompt
    prompt_template = """다음은 문서(CONTEXT) 중 일부 내용입니다. 이 문서 목록을 기반으로 주요 내용을 한국어(Korean)로 요약해:
    너는 다음과 같은 규칙을 따라야돼.
    - 특수기호 제외
    - 문장에 대한 연관성이 있음
    - 핵심 내용에 대한 언급
    - 글자만 생성해
    CONTEXT:
    {context}
    
    답변:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # 프롬프트와 언어 모델을 연결하여 체인을 생성합니다. gpu_llm.bind(stop=["\n\n"])
    #map_chain = prompt | gpu_llm | StrOutputParser()
    map_chain = LLMChain(llm=llm, prompt=prompt)

    reduce_template = """다음은 요약의 집합입니다. 이것(CONTEXT)들을 바탕으로 통합된 요약을 자연스러운 문장으로 한국말(Korean)로 만들어.
    너는 다음과 같은 규칙을 따라야돼.
    - 특수기호 제외
    - 문장에 대한 연관성이 있음
    - 핵심 내용에 대한 언급
    - 글자만 생성해
    CONTEXT:
    {context}
    
    답변:"""

    # Reduce 프롬프트 완성
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Reduce에서 수행할 LLMChain 정의
    #reduce_chain = reduce_prompt | gpu_llm | StrOutputParser()
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    

    
    # 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_prompt=document_prompt,
        document_variable_name="context"
    )


    # Map 문서를 통합하고 순차적으로 Reduce합니다.
    reduce_documents_chain = ReduceDocumentsChain(
        # 호출되는 최종 체인입니다.
        combine_documents_chain=combine_documents_chain,
        # # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
        collapse_documents_chain=combine_documents_chain,
        # # 문서를 그룹화할 때의 토큰 최대 개수입니다.
        token_max=4000,
    )

    # ========== ⑤ Map-Reduce 통합단계 ========== #

    # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
    map_reduce_chain = MapReduceDocumentsChain(
        # Map 체인
        llm_chain=map_chain,
        # Reduce 체인
        reduce_documents_chain=reduce_documents_chain,
        # # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
        document_variable_name="context",
        # # 출력에서 매핑 단계의 결과를 반환합니다.
        # return_intermediate_steps=False,
    )


    
    result = map_reduce_chain.invoke({"input_documents":splits}, return_only_outputs=True)
    print(result)
    with open('data/summarization/example.json', 'w', encoding='utf-8') as file:
    # 변수의 내용을 파일에 쓰기
        json.dump(result, file, default=str)

    return result
summarization()