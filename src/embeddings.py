from langchain_community.embeddings import LlamaCppEmbeddings
from logger import get_logger

logger = get_logger("embeddings", "DEBUG")

class Embedder(LlamaCppEmbeddings):
    def __init__(self, model_path: str):
        super().__init__(model_path=model_path)
        logger.info(f"Instantiate Embedder with model: {model_path}")
