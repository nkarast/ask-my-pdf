from langchain_community.document_loaders import PyPDFLoader
from embeddings import Embedder
from vector_store import KnowledgeBase
from pathlib import Path
from logger import get_logger

logger = get_logger("prepare_data", "WARN")

class DataLoader(PyPDFLoader):
    """Loads the PDF data"""
    #def __init__(self, file_path: Union[list, str]):
    def __init__(self, filename: str, embedding_model_path: str):
        self.asset_path = Path('asset')
        # parent -> src + parent -> root
        self.root_path = Path(__file__).parent.parent.resolve()
        self.file_to_load = str((self.root_path / self.asset_path / filename))
        logger.info(f"Loading file: {self.file_to_load}")

        super().__init__(self.file_to_load)
        self.embedding_model_path = embedding_model_path

        logger.info("Splitting document")
        self.pages = self.get_pages()


        logger.info("Loading embedding model")
        self.embedder = self.get_embedder()

        logger.info("Building vector store")
        self.docs = self.get_knowledge_base()

        logger.info("Preparing the retriever")
        self.retriever = self.prepare_retriever()

    def get_pages(self):
        return self.load_and_split()
    
    def get_embedder(self):
        return Embedder(self.embedding_model_path)
    
    def get_knowledge_base(self):
        return KnowledgeBase(self.pages, self.embedder).build_kb()
    
    def prepare_retriever(self):
        retriever = self.docs.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k": 4,
                "fetch_k": 20
            }
        )
        return retriever
    
    def test_retriever(self, question: str):
        return self.retriever.get_relevant_documents(question)
