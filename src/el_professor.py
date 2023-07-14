# Path: RetrievalQA_MMR.py
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.vectorstores import FAISS
import tiktoken
try:
    from src.processing import (
        TOKEN_LIMIT, Dict, List, Document, 
        get_pdfs, extract_text_loaders, text_splitter, initialise_llms, extract_text_from_docs
        )
    from src.RetrievalQA_mod import RetrievalQA
except ModuleNotFoundError:
    from processing import *
    from RetrievalQA_mod import RetrievalQA

# TO-DOS:
# - Implement Conversational RetrievalQA

load_dotenv()

def _make_fetch_size(docstore: FAISS) -> int:
    size = docstore.index.ntotal
    return size

def _token_limiter(docstore: FAISS, question: str, fetch_size: int, _k: int) -> FAISS | bool:
    docs = docstore.similarity_search(question, k = _k, fetch_k = fetch_size)
    text = extract_text_from_docs(docs)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if TOKEN_LIMIT > len(tokens):
        return docstore
    return False

def create_vdb(foldername: str, embed_type: Embeddings) -> FAISS:
    loaders = get_pdfs(foldername)
    text = extract_text_loaders(loaders)
    chunks = text_splitter(text, docs = True)
    docstore = FAISS.from_documents(documents = chunks, embedding = embed_type)
    return docstore

def create_vdb_from_txts(foldername: str, embed_type: Embeddings) -> FAISS:
    folderpath = f"./Notes/{foldername}"
    bank = []
    for file in os.listdir(folderpath):
        if file.endswith(".txt"):
            with open(f"{folderpath}/{file}", "r") as f:
                text = f.read()
                bank.append(text)
    docstore = FAISS.from_texts(texts = bank, embedding = embed_type)
    return docstore

def save_vdb(docstore: FAISS, foldername: str):
    docstore.save_local(f"./VectorDBs/{foldername}.faiss")

def choose_vdb(foldername: str, embed_type: Embeddings) -> FAISS:
    try:
        # docstore = FAISS.load_local(f"./VectorDBs/{foldername}.faiss", hembeddings)
        docstore = FAISS.load_local(f"./VectorDBs/{foldername}.faiss", embeddings = embed_type)
        print("Loaded from file")
        return docstore
    except:
        docstore = create_vdb(foldername, embed_type)
        print("Created new knowledge base")
        return docstore

def embed_type_chooser(embed_type: str, api_key: str = os.getenv("OPENAI_API_KEY")) -> Embeddings:
    if embed_type in ["h", "H", "hypo"]:
        llm = OpenAI(temperature = 0, openai_api_key = api_key)
        hembeddings = HypotheticalDocumentEmbedder.from_llm(llm=llm, base_embeddings = base_embeddings, prompt_key = "web_search")
        return hembeddings
    if embed_type in ["o", "O", "openai"]:
        base_embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        base_embeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
        return base_embeddings
    raise ValueError("Invalid Embedding Type: Choose 'h' for Hypothetical Embeddings or 'o' for OpenAI Embeddings.")

def _make_prompt_assist():
    system_template = """You will be provided with a chunk of context information, delimited by triple backticks. Use the context and your prior knowledge to understand the content fully.
    Your job is to answer the question that is based around the content of the given context. You may use your prior knowledge to explain further any ideas relevant to the qustion that are found in the context.
    If the context does not provide a direct answer to the question, use reasonable judgement to infer an answer from the context using your prior knowledge.
    For any equations that are present in your answer, make sure to clearly explain what each term means and include an explaination of any relevant keywords and ideas.
    Take time to make sure all equations generated in the answer are mathematically and scientifically sound.
    The output of your answer should follow the specified output format.
    ----------------
    Output Format: The context may contain missing or corrupted characters and symbols. If you are able to determine the appropriate unicode character, include it in the generated answer. Otherwise, do not include it in the generated answer.
    e.g. if the context contains "x^2", and you are able to determine that the appropriate unicode character is "²", then the generated answer should contain "x²".
    If the output contains equations, assuming you have extensive prior knowledge about said equations, you should rewrite the equation, adhering to the specified output format and making sure any missing or corrupted characters are replaced with the appropriate character.
    Provide the answer in markdown format and make sure ALL equations are in a printable format.
    ----------------
    Context: ```{context}```"""
    messages = [
        HumanMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    # system_template = """Task: Use the following pieces of context, delimited by triple backticks, to answer the question at the end. Do not make up an answer that cannot be inferred from the provided context.
    # If the provided context does not relate to the question, do not make up an answer and instead say that the context does not provide an answer. However, you may use your prior knowledge to explain further any ideas relevant to the qustion that are found in the context.
    # Make sure to clearly explain what each term means in equations and include an explaination of any relevant keywords and ideas.
    # Take time to make sure all equations generated in the answer are mathematically and scientifically sound.
    # The output should follow the specified output format.
    # ----------------
    # Output Format: If the context contains 'cid:xxx' where xxx is a number, do not include the 'cid:xxx' in the generated answer but use your best judgement and prior knowledge to replace it with the appropriate and relevant unicode character.
    # If you are not able to determine the appropriate unicode character, do not include it in the generated answer.
    # If the output contains equations, assuming you have extensive prior knowledge about said equations, you should rewrite the equation, adhering to the specified output format.
    # Provide the answer in markdown format and make sure ALL equations are in a printable format.
    # ----------------
    # Context: ```{context}```"""
    # messages = [
    #     SystemMessagePromptTemplate.from_template(system_template),
    #     HumanMessagePromptTemplate.from_template("{question}"),
    # ]
    final_prompt = ChatPromptTemplate.from_messages(messages)
    return final_prompt

def answer_question(model, docstore: FAISS, question: str, speed: int = 0.52) -> Dict[str, List[Document]]:
    fetch_size = _make_fetch_size(docstore)
    _k = round(fetch_size**speed)
    docstore_trimed = _token_limiter(docstore, question, fetch_size, _k)
    while not docstore_trimed:
        speed -= 0.025
        _k = round(fetch_size**speed)
        docstore_trimed = _token_limiter(docstore, question, fetch_size, _k)
    PROMPT = _make_prompt_assist()
    qachain = load_qa_chain(llm = model, chain_type = "stuff", prompt = PROMPT)
    rqa = RetrievalQA(combine_documents_chain = qachain,
                      docstore = docstore_trimed,
                      return_source_documents = True,
                      k = _k,
                      fetch_size = fetch_size)
    answer_and_sources = rqa({"query": question})
    print("Answer generated")
    # print(answer_and_sources["source_documents"])
    return answer_and_sources

def get_sources(answer_and_sources: Dict[str, List[Document]]) -> str:
    sources = ""
    for source in answer_and_sources["source_documents"]:
        sources += source.page_content
    sources = " ".join(sources.split())
    return sources

def regen_answer(answer: str, detail: int, model) -> str:
    system_template = """Your job is to use your own knowledge and best judgement to increase the level of detail in the provided answer.
    The detail level is out of 10 and the higher the number, the more detail you should add. For reference, detail level of 10 should be the most detailed answer possible including comprehensive explainations of all relevant ideas and concepts.
    Make sure to explain what each term means in equations and include an explaination of any relevant keywords and ideas.
    You have to make any equations generated in the answer are mathematically sound.
    Provide the answer in markdown format and make sure ALL equations are in LATEX format.
    Detail level: {detail}
    ----------------
    {answer}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = LLMChain(llm=model, prompt=prompt)
    answer = chain.run(detail = detail, answer = answer)
    return answer


# Main loop
if __name__ == "__main__":
    
    chat3_5, chat4 = initialise_llms()
    foldername = input("Enter the name of the subject: ").casefold()
    # embed_type = input("Enter the type of embeddings you want to use: ")
    embedder = embed_type_chooser("o")
    docstore = create_vdb_from_txts(foldername, embedder)
    save_vdb(docstore, foldername)
    
    while True:
        question = input("Enter your question: ")
        result = answer_question(chat4, docstore, question)
        sources = get_sources(result)
        print(f"Answer: {result['result']} \n\nBackground Information: {sources} \n")
        # os.system(f"say '{answer['result']}' --rate=179.4")
