from langchain_community.llms import LlamaCpp
from logger import get_logger

logger = get_logger("llm", "DEBUG")


class Llm(LlamaCpp):
    """Wrapper around `langchain_community.llms.LlamaCpp`
    The model is initialized with a local `model_path` and by default has
       - context length: 4000
       - runs on GPU (n_gpu_layers = -1)
       - temperature = 0
       - verbose = False
    """

    def __init__(self, model_path: str):
        """Instantiates a Llm model wrapping the LlamaCpp

        Args:
            model_path (str): Local path to the .bin/.gguf holding model
        """
        super().__init__(
            model_path=model_path,
            verbose=False,
            n_ctx=4000,
            n_gpu_layers=-1,
            temperature=0.1,
        )
