import gradio as gr
from prepare_data import DataLoader
from rag_chain import RAGChain
from llm import Llm
from logger import get_logger

logger = get_logger("main", "DEBUG")


MODEL_PATH = "/Users/nkarast/Documents/data-science/llama/models/llama-2-7b-chat.Q4_K_M.gguf"
PDF_PATH = "CERN-ACC-NOTE-2019-0046.pdf"
logger.info('Starting')
logger.info("Preparing data store...")
data = DataLoader(filename=PDF_PATH, embedding_model_path=MODEL_PATH)

logger.info("Setting up model")
llm = Llm(model_path=MODEL_PATH)

logger.info("Preparing chain")
chain = RAGChain(data.retriever, data.docs, llm)

def ask(question):
    return chain.run_chain(question)

demo = gr.Interface(
    fn = ask,
    inputs='textbox',
    outputs='textbox'
)

demo.launch(inbrowser=True)