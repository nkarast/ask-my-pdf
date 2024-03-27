from langchain_community.llms import LlamaCpp
from logger import get_logger

logger = get_logger("llm", "DEBUG")

class Llm(LlamaCpp):
    def __init__(self, model_path: str):
        super().__init__(model_path=model_path,
                         n_ctx = 4000,
                         n_gpu_layers=-1,
                         temperature = 0.0,
                         verbose=False)
