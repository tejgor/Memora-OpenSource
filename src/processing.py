import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

TOKEN_LIMIT = 2000
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100
REQ_TIMEOUT = 200

def initialise_llms():
    from langchain.chat_models import ChatOpenAI
    load_dotenv()
    chat3_5 = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo", request_timeout = REQ_TIMEOUT)
    chat4 = ChatOpenAI(temperature = 0, model_name = "gpt-4", request_timeout = REQ_TIMEOUT)
    chat3_5.openai_api_key = os.getenv("OPENAI_API_KEY")
    chat4.openai_api_key = os.getenv("OPENAI_API_KEY")
    return chat3_5, chat4

def initialise_llms_with_key(api_key: str):
    from langchain.chat_models import ChatOpenAI
    load_dotenv()
    chat3_5 = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo", openai_api_key = api_key, request_timeout = REQ_TIMEOUT)
    chat4 = ChatOpenAI(temperature = 0, model_name = "gpt-4", openai_api_key = api_key, request_timeout = REQ_TIMEOUT)
    return chat3_5, chat4

def extract_text_from_docs(docs: List[Document]) -> str:
    text = []
    for page in docs:
        content = page.page_content
        text.append(content)
    text = " ".join(text)
    return text

def text_splitter(text: str, docs: bool = False, chnk_size: int = CHUNK_SIZE) -> List[str] | List[Document]:
    # splitter = NLTKTextSplitter(separator = ".",chunk_size = chnk_size, chunk_overlap = CHUNK_OVERLAP)
    # splitter = CharacterTextSplitter(separator = "\n", chunk_size = chnk_size, chunk_overlap = CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(separators = [" ",",","\n"], chunk_size = chnk_size, chunk_overlap = CHUNK_OVERLAP)
    if docs:
        chunks = splitter.create_documents([text])
        return chunks
    chunks = splitter.split_text(text)
    return chunks

def get_pdfs(foldername: str):
    folderpath = f"./Notes/{foldername}"
    loaders = [PyPDFLoader(os.path.join(folderpath, fn)) for fn in os.listdir(folderpath)]
    return loaders

def extract_text_loaders(loaders) -> str:
    text = []
    for loader in loaders:
        doc = loader.load()
        for page in doc:
            text.append(page.page_content)
    text = " ".join(text)
    return text

if __name__ == "__main__":
    foldername = input("Enter folder name: ")
    loaders = get_pdfs(foldername)
    text = extract_text_loaders(loaders)
    
    print(text)
    