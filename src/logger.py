import logging
from typing import Union

def get_logger(name:str, level: Union[int, str]) -> logging.Logger:
    logger = logging.getLogger("rag.app."+name)

    log_level = level
    if isinstance(name, str):
        log_level = logging.getLevelName(level.upper())
    
    logger.setLevel(log_level)
    
    ch = logging.StreamHandler()

    format = '%(asctime)s - %(levelname)-5s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt=format)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
