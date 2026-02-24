import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
# from langchain_community.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))

chat_history = InMemoryChatMessageHistory()

# upload pdf file
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# chops the texts
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50 
        # تداخل بين القطع لعدم فقدان السياق
    )
    chunks = splitter.split_documents(documents)
    return chunks

# create vector store
def create_vectorstore(chunks):
    # embedding each word has a series of numbers
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# load vector store
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

# build the chain (lainchain)
def build_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # free and fast
        temperature=0,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferMemory(
        chat_memory = chat_history,
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return chain