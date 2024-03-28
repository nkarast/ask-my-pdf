from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from logger import get_logger

logger = get_logger("rag_chain", "DEBUG")


class RAGChain:
    """A wrapper around a RAG Chain"""

    def __init__(self, retriever, docs, llm):
        """Instantiate the class given a retriever, the list of pages
        extracted an an LLM to generate the answer

        Args:
            retriever (langchain_core.vectorstores.VectorStoreRetriever): _description_
            docs (langchain_community.vectorstores.chroma.Chroma): _description_
            llm (): _description_
        """
        self.prompt = self.get_prompt()
        self.retriever = retriever
        self.docs = docs
        self.llm = llm
        logger.info("Building chain")
        self.chain = self.create_chain()

    def format_docs(self, docs) -> str:
        """Formatting doc content for the context of the prompt

        Args:
            docs (list[]): List of documents. Each document is a selected page from the PDF.

        Returns:
            str: Returns the string of the page content formatted with empty lines between them
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def get_prompt(self) -> str:
        """Create the prompt for the RAG. Modifying `rlm/rag-prompt` for Llama model.

        Returns:
            str: _description_
        """
        template = """[INST]You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

<question>
Question: {question} 
</question>

<context>
Context: {context} 
</context>

Answer:[/INST]"""
        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def create_chain(self):
        """Creates the langchain_core.chains.LLMChain runnable

        Returns:
            (langchain_core.runnables.base.RunnableSequence): The LLM Chain runnable
        """
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def run_chain(self, question: str) -> str:
        """Invokes the runnable of the chain with the user input

        Args:
            question (str): User input to be added in the template for the LLM Chain

        Returns:
            str: Output of the LLM's RAG
        """
        return self.chain.invoke(question)
