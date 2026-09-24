"""LangChain retriever wrapping the high-speed bm25s library (sparse keyword search)."""
from typing import List

import bm25s
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class LangchainBM25sRetriever(BaseRetriever):
    """Custom LangChain retriever wrapper around the high-speed bm25s library."""

    documents: list[Document]
    retriever_index: bm25s.BM25
    k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 4, method: str = "atire"):
        if not documents:
            raise ValueError("Document list passed to BM25s index is empty.")
        corpus = [doc.page_content for doc in documents]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever_index = bm25s.BM25(method=method)
        retriever_index.index(corpus_tokens)
        return cls(documents=documents, retriever_index=retriever_index, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        query_tokens = bm25s.tokenize(query, stopwords="en")
        results, scores = self.retriever_index.retrieve(query_tokens, k=self.k, show_progress=False)
        retrieved: list[Document] = []
        for idx, score in zip(results[0], scores[0]):
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = float(score)
            retrieved.append(doc)
        return retrieved
