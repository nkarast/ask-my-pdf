from langchain_community.embeddings import LlamaCppEmbeddings
from logger import get_logger

logger = get_logger("embeddings", "DEBUG")


class Embedder(LlamaCppEmbeddings):
    """Wrapper around the `langchain_community.embeddings.LlamaCppEmbeddings`"""

    def __init__(self, model_path: str):
        """Initialize the embedding model

        Args:
            model_path (str): Full path to the embedding model (gguf, bin, ggml)
        """
        super().__init__(model_path=model_path, verbose=False)
        logger.info(f"Instantiate Embedder with model: {model_path}")
