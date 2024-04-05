from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from logger import get_logger

logger = get_logger("vector_store", "DEBUG")


class KnowledgeBase:
    """Creates a Chroma vector store in memory"""

    def __init__(self, pages: list, embeddings: LlamaCppEmbeddings):
        """Instantiate a KnowledgeBase built on ChromaDB.

        Args:
            pages (list[angchain_core.documents.base.Document]): _description_
            embeddings (langchain_community.embeddings.LlamaCppEmbeddings): _description_
        """
        self.pages = pages
        self.embeddings = embeddings

    def build_kb(self) -> Chroma:
        """Build the KB from the documents

        Returns:
            langchain_community.vectorstores.chroma.Chroma: The built ChromaDB
        """
        return Chroma.from_documents(self.pages, self.embeddings)
