from langchain_community.document_loaders import PyPDFLoader
from embeddings import Embedder
from vector_store import KnowledgeBase
from pathlib import Path
from logger import get_logger

logger = get_logger("prepare_data", "INFO")

class DataLoader(PyPDFLoader):
    """Loads the PDF data"""

    def __init__(self, filename: str, embedding_model_path: str):
        """Initializes the data loader with a filename to a PDF and 
        the local path to the embedding model.

        Args:
            filename (str): Name of the PDF. Assumed to be under a `asset/` 
                            folder which sits at the same level to `src`
            embedding_model_path (str): Path to the embedding model
        """
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
        """Split the document into pages. Builds a LC document per page.

        Returns:
            list[langchain_core.documents.base.Document]: List of PDF pages in the format of langchain_core.documents.base.Document
        """
        return self.load_and_split()
    

    def get_embedder(self):
        """Get the embedding model

        Returns:
            (Embedder): Get an instatiantion of the Embedder class
        """        
        return Embedder(self.embedding_model_path)
    

    def get_knowledge_base(self):
        """Instantiate and build the knowledge base.

        Returns:
            (KnowledgeBase): Returns the built knowledge base based on the pages extracted and the Embedder
        """
        return KnowledgeBase(self.pages, self.embedder).build_kb()
    

    def prepare_retriever(self):
        """Prepares the knowledge base as a Retriever for the application

        Returns:
            (langchain_core.vectorstores.VectorStoreRetriever): The Runanble that can be passed to the chain to retrieve the documents
        """
        retriever = self.docs.as_retriever(
            search_type = "mmr",
            search_kwargs = {
                "k": 4,
                "fetch_k": 20
            }
        )
        return retriever
    
    
    def test_retriever(self, question: str):
        """Test the retriever based on a user question.

        Args:
            question (str): User question to test the retriever with

        Returns:
            (langchain_community.vectorstores.chroma.Chroma): List of documents from the KnowledgeBase retrieved based on the user question
        """
        return self.retriever.get_relevant_documents(question)
