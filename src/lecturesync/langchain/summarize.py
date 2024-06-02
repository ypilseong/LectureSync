from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain
import os

from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ./cache/ 경로에 다운로드 받도록 설정
os.environ["TRANSFORMERS_CACHE"] = "./data/llm_model/"
os.environ["HF_HOME"] = "./data/llm_model/"

def summarization():
    tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B-Instruct-preview",
                                              padding_side='left',
                                              truncation=True)
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

    

    loader = PyPDFLoader('data/doc_data/머신러닝을활용한가짜리뷰탐지연구사용자행동분석을중심으로.pdf')
    docs = loader.load()
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",  # 분할기준
        chunk_size=1000,   # 사이즈
        chunk_overlap=50, # 중첩 사이즈
    )
    splits = text_splitter.split_documents(docs)
    
    print(f'총 분할된 도큐먼트 수: {len(splits)}')

    # Define prompt
    prompt_template = """다음은 문서 중 일부 내용입니다.
    {pages}
    이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
    답변:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # 프롬프트와 언어 모델을 연결하여 체인을 생성합니다. gpu_llm.bind(stop=["\n\n"])
    #map_chain = prompt | gpu_llm | StrOutputParser()
    map_chain = LLMChain(llm=gpu_llm, prompt=prompt)

    reduce_template = """다음은 요약의 집합입니다:
    {doc_summaries}
    이것들을 바탕으로 통합된 요약을 만들어 주세요.
    답변:"""

    # Reduce 프롬프트 완성
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Reduce에서 수행할 LLMChain 정의
    #reduce_chain = reduce_prompt | gpu_llm | StrOutputParser()
    reduce_chain = LLMChain(llm=gpu_llm, prompt=reduce_prompt)
    

    document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
    )
    # 문서의 목록을 받아들여, 이를 단일 문자열로 결합하고, 이를 LLMChain에 전달합니다.
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_prompt=document_prompt,
        document_variable_name="doc_summaries"
    )


    # Map 문서를 통합하고 순차적으로 Reduce합니다.
    reduce_documents_chain = ReduceDocumentsChain(
        # 호출되는 최종 체인입니다.
        combine_documents_chain=combine_documents_chain,
        # 문서가 `StuffDocumentsChain`의 컨텍스트를 초과하는 경우
        collapse_documents_chain=combine_documents_chain,
        # 문서를 그룹화할 때의 토큰 최대 개수입니다.
        token_max=4000,
    )

    # ========== ⑤ Map-Reduce 통합단계 ========== #

    # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합합니다.
    map_reduce_chain = MapReduceDocumentsChain(
        # Map 체인
        llm_chain=map_chain,
        # Reduce 체인
        reduce_documents_chain=reduce_documents_chain,
        # 문서를 넣을 llm_chain의 변수 이름(map_template 에 정의된 변수명)
        document_variable_name="pages",
        # 출력에서 매핑 단계의 결과를 반환합니다.
        return_intermediate_steps=False,
    )

    # # Define StuffDocumentsChain
    # summaries=[]
    # for split in splits:
    #     response = gpu_chain.invoke({"text": split.page_content})
    #     summaries.append(response)
    #     print(response)
    
    # final_summary = "\n\n".join(summaries)
    # print(final_summary)    

    # with open('data/summariztion/example.txt', 'w', encoding='utf-8') as file:
    # # 변수의 내용을 파일에 쓰기
    #     file.write(final_summary)

    # return response
    result = map_reduce_chain.run(splits)
    print(result)

summarization()
