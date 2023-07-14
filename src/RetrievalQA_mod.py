from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import Field
from langchain.vectorstores import FAISS
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import BaseRetrievalQA


class RetrievalQA(BaseRetrievalQA):
    fetch_size: int = 100000
    k: int

    docstore: FAISS = Field(exclude=True)
    
    def _get_docs(self, question: str) -> List[Document]:
        return self.docstore.similarity_search(question, k = self.k, fetch_k = self.fetch_size)
    
    def _aget_docs(self, question: str) -> List[Document]:
        return self.docstore.similarity_search(question, k = self.k, fetch_k = self.fetch_size)




#---------- Legacy Imports ----------#

# from __future__ import annotations
# from abc import abstractmethod
# from typing import Any, Dict, List, Optional
# from pydantic import Extra, Field
# from langchain.vectorstores import FAISS
# from langchain.chains.base import Chain
# from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.chains.llm import LLMChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
# from langchain.prompts import PromptTemplate
# from langchain.schema import BaseLanguageModel, Document
# from langchain.chains.retrieval_qa.base import BaseRetrievalQA