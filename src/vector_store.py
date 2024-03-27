from langchain_community.vectorstores import Chroma
from logger import get_logger

logger = get_logger("vector_store", "DEBUG")

class KnowledgeBase():
    def __init__(self, pages, embeddings):
        self.pages = pages
        self.embeddings = embeddings

    def build_kb(self):
        return Chroma.from_documents(self.pages, self.embeddings)