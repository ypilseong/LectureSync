import os
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI

os.environ["OPNEAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

class DocumentSummarizer:
    def __init__(self,
                 pdf_path=None,
                 txt_path=None,
                 model_url='http://172.16.229.33:11436',
                 model_name='EEVE-Korean-Instruct-10.8B',
                 temperature=0.3,
                 chunk_size=2000,
                 chunk_overlap=200):
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self.model_url = model_url
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 설정된 환경 변수
        os.environ["TRANSFORMERS_CACHE"] = "./data/llm_model/"
        os.environ["HF_HOME"] = "./data/llm_model/"

    def load_documents(self):
        docs = []
        if self.pdf_path:
            for file_path in self.pdf_path:
                pdf_loader = PyPDFLoader(file_path)
                docs.extend(pdf_loader.load())
        
        if self.txt_path:
            for file_path in self.txt_path:
                txt_loader = TextLoader(file_path)
                docs.extend(txt_loader.load())
        return docs

    def split_documents(self, docs):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(docs)

    def create_llm(self):
        # return ChatOllama(base_url=self.model_url,
        #               model=self.model_name,
        #               temperature=self.temperature)
        return ChatOpenAI(temperature=0.1,
                          model_name='gpt-4o')

    def create_map_reduce_chain(self, llm):
        # Define map prompt
        map_prompt_template = """다음은 문서(CONTEXT) 중 일부 내용입니다. 이 문서 목록을 기반으로 주요 내용을 한국어(Korean)로 요약해:
        너는 다음과 같은 규칙을 따라야돼.
        - 특수기호 제외
        - 문장에 대한 연관성이 있음
        - 핵심 내용에 대한 언급
        - 글자만 생성해
        - 최대한 많은 정보를 포함해
        CONTEXT:
        {map}
        
        답변:"""
        map_prompt = PromptTemplate.from_template(map_prompt_template)

        # Define reduce prompt
        reduce_template = """다음은 요약의 집합입니다. 이것(CONTEXT)들을 바탕으로 통합된 요약을 자연스러운 문장으로 한국말(Korean)로 만들어.
        너는 다음과 같은 규칙을 따라야돼.
        - 특수기호 제외
        - 문장에 대한 연관성이 있음
        - 핵심 내용에 대한 언급
        - 글자만 생성해
        - 최대한 많은 정보를 포함해
        CONTEXT:
        {pages}
        
        답변:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        # Define chains
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
        
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
        
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=map_chain,
            document_prompt=document_prompt,
            document_variable_name="map"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=8000,
        )

        return MapReduceDocumentsChain(
            llm_chain=reduce_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="pages",
        )
    def create_stuff_chain(self, llm):
        stuff_template = """다음 문서(CONTEXT)를 기반으로 주요 내용을 한국어(Korean)로 요약해:
        너는 다음과 같은 규칙을 따라야돼.
        - 특수기호 제외
        - 문장에 대한 연관성이 있음
        - 핵심 내용에 대한 언급
        - 글자만 생성
        - 최대한 많은 정보를 포함
        CONTEXT:
        {pages}
        
        답변:"""
        stuff_prompt = PromptTemplate.from_template(stuff_template)
        stuff_chain = stuff_prompt | llm | StrOutputParser()
        
        return stuff_chain
        
    def summarize(self, type=None):
        docs = self.load_documents()
        splits = self.split_documents(docs)
        print(f'총 분할된 도큐먼트 수: {len(splits)}')

        llm = self.create_llm()
        if type == 'stuff':
            stuff_chain = self.create_stuff_chain(llm)
            result = stuff_chain.invoke({"pages":splits})
            print(result)
            return result
        
        if type == 'map_reduce':
            map_reduce_chain = self.create_map_reduce_chain(llm)
            result = map_reduce_chain.invoke({"input_documents":splits}, return_only_outputs=True)
            print(result)
            return result['output_text']
        