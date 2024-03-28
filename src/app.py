from dotenv import dotenv_values
import pathlib
from collections import OrderedDict
from logging import Logger
import gradio as gr
from prepare_data import DataLoader
from rag_chain import RAGChain
from llm import Llm
from logger import get_logger

logger = get_logger("main", "DEBUG")


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

    logger.info("Setting up model")
    llm = Llm(model_path=model_path)

    logger.info("Preparing chain")
    chain = RAGChain(data.retriever, data.docs, llm)
    return chain


if __name__ == "__main__":
    conf = read_env()
    print(conf)
    chain = initialize_chain(conf, logger)

    def ask_my_pdf(question: str) -> str:
        """Function to wrap the interface button around in order to invoke the chian

        Args:
            question (str): User question

        Returns:
            str: LLM output
        """
        return chain.run_chain(question)

    def __ask_my_pdf(question: str) -> str:
        """Dummy question for testing without passing through LLM

        Args:
            question (str): User input

        Returns:
            str: Same as user input (passthrough)
        """
        return question

    demo = gr.Interface(fn=__ask_my_pdf, inputs="textbox", outputs="textbox")

    demo.launch(inbrowser=True)
