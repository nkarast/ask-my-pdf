from dotenv import dotenv_values
import pathlib
from collections import OrderedDict
from logging import Logger

# import gradio as gr
import streamlit as st
from prepare_data import DataLoader
from rag_chain import RAGChain
from llm import Llm
from logger import get_logger

logger = get_logger("main", "DEBUG")
data = None
chain = None
llm = None


def read_env(env: str = ".env") -> OrderedDict:
    """Read the .env values.

    Args:
        env (str, optional): The config file to load. Defaults to "./env".

    Returns:
        OrderedDict: The values in the dict format
    """
    cur = str(pathlib.Path(__file__).parent.resolve())
    return dotenv_values("/".join([cur, env]))


def _get_local_model_name(env: OrderedDict) -> str:
    """Return the full model name from the .env

    Args:
        env (OrderedDict): an OrderedDict from dotenv_values

    Returns:
        str: The full local path for the model
    """
    return env["MODEL_NAME"]


def _get_local_asset_path(env: OrderedDict) -> str:
    """Return the path for the local asset directory from the .env

    Args:
        env (OrderedDict): an OrderedDict from dotenv_values

    Returns:
        str: The local path for the `asset` dir
    """
    return env["PDF_PATH"]


def _get_local_pdf(env: OrderedDict) -> str:
    """Returns the full local filename for testing purposes

    Args:
        env (OrderedDict): n OrderedDict from dotenv_values

    Returns:
        str: The full local pdf path and filename
    """
    return "/".join([_get_local_asset_path(env), env["TEMP_PDF"]])


def initialize_chain(
    env: OrderedDict, logger: Logger, filename: str = None
) -> RAGChain:
    """Initialize the setup for the chain

    Args:
        env (OrderedDict): The env file for local environment variables
        logger (Logger): The main app logger
        filename (str, optional): Name of the PDF to be loaded.

    Returns:
        RAGChain: Runnable chain to wrap on a UI accessible function
    """

    model_path = _get_local_model_name(env)
    if filename is None:
        filename = _get_local_pdf(env)
    else:
        filename = "/".join([_get_local_asset_path(env), filename])

    logger.info(f"Using file : {filename}")

    logger.info("Starting")
    logger.info("Preparing data store...")
    data = DataLoader(filename=filename, embedding_model_path=model_path)
    st.session_state.data = False

    if not st.session_state.data:
        st.session_state.data = data

    logger.info("Setting up model")
    llm = Llm(model_path=model_path)
    st.session_state.llm = False
    if not st.session_state.llm:
        st.session_state.llm = llm

    logger.info("Preparing chain")
    st.session_state.chain = False
    chain = RAGChain(data.retriever, data.docs, llm)
    if not st.session_state.chain:
        st.session_state.chain = chain


#    return chain


def run_chain(chain, query):
    return chain.run_chain(query)


### MAIN

conf = read_env()
st.set_page_config(page_title="AskMyPDF", page_icon=":sunglasses:")

st.title("Ask my PDF")

st.markdown(
    """
            | Document Title | Model Name |
            | -- | -- | 
            | {doc} | {model}|
            """.format(doc=conf["TEMP_PDF"], model=conf["MODEL_NAME"].split("/")[-1])
)
st.text("")


but_press = st.button(
    "Click to Process Document", on_click=initialize_chain, args=(conf, logger)
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message("human").write(message[0])
    st.chat_message("ai").write(message[1])

if query := st.chat_input():
    st.chat_message("human").write(query)
    response = run_chain(st.session_state.chain, query)
    st.chat_message("ai").write(response)
