from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def ingest_tweets_locally():
    # 1. Load and Chunk
    print('starting to ingest tweets...')
    loader = TextLoader("tweets.txt") 
    print('loading tweets...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(loader.load())
    print('tweets loaded and split into chunks')
    # 2. Local Embeddings (Running on your 5060 Ti)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    print('embeddings model loaded')
    # 3. Store in Local DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./local_tweet_db"
    )
    print("Done! Tweets indexed locally.")

if __name__ == "__main__":
    print('starting the ingest_tweets_locally function...')
    ingest_tweets_locally()