import logging
from rich.logging import RichHandler
from logging.config import dictConfig

class NewRichHandler(RichHandler):
    KEYWORDS = {
        'size',
        'Epoch',
        'Batch',
        'Img',
        'Loss',
        'fg',
        'bg',
        'pos',
        'neg',
        'iou',
        'cls',
        'Loss_S',
        'Loss_R',
        'Loss_L',
        'LR',
        '_'     
    }

class Logger(object):
    def __init__(self,log_file_name,log_level,logger_name):
        # firstly, create a logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        # secondly, create a handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = NewRichHandler(rich_tracebacks=True, tracebacks_show_locals=True)
        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s'
        )
        rich_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(rich_formatter)
        # finally, add the Hander to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

if __name__ == "__main__":
    logger = Logger('./log.txt', logging.DEBUG, 'demo').get_log()
    logger.info('hello')