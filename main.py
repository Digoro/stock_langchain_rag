import yfinance as yf
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
os.environ["GOOGLE_API_KEY"] = ""

# 데이터 수집
def getStockNews(ticker):
    ticker = yf.Ticker(ticker)
    return ticker.news
def getStockPrice(ticker):
    ticker = yf.Ticker(ticker)
    return ticker.history(period="1mo")
def getStockFinances(ticker):
    ticker = yf.Ticker(ticker)
    return ticker.financials

stockCode = "META"
news = getStockNews(stockCode)
price = getStockPrice(stockCode)
finance = getStockFinances(stockCode)


# 데이터 스플릿
news_list = []
for new in news:
  news_list.append(new.get("link"))
print(news_list)

loader = WebBaseLoader(
     web_paths=(news_list),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(len(splits))


# 데이터 임베딩
hf = HuggingFaceEmbeddings(
    model_name='cointegrated/rubert-tiny2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)
vectorstore = FAISS.from_documents(documents=splits, embedding=hf)
retriever = vectorstore.as_retriever()


# AI 모델 및 프롬프트 생성
template = f"""다음 최신 정보를 참고해서 답변해줘
관련 뉴스: {{news}}

종목 주가:
{str(price)}
      
종목 재무재표:
{str(finance)} 

질문: {{question}}

답변: """
prompt = ChatPromptTemplate.from_template(template)
model = ChatGoogleGenerativeAI(model="gemini-pro")


# 체이닝 생성 및 실행
chain = (
    {
        "news": retriever, 
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("META의 재무재표와 주가와 뉴스를 기준으로 투자 포트폴리오 작성해줘")