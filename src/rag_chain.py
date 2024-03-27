from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from logger import get_logger

logger = get_logger("rag_chain", "DEBUG")

class RAGChain():
    def __init__(self, retriever, docs, llm):
        self.prompt = hub.pull("rlm/rag-prompt")
        self.retriever = retriever
        self.docs = docs
        self.llm = llm
        logger.info("Building chain")
        self.chain = self.create_chain()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self):
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm 
            | StrOutputParser()
            
        )
        return rag_chain
    
    def run_chain(self, question: str) -> str:
        return self.chain.invoke(question)

