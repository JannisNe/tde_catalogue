import logging
main_logger = logging.getLogger(__name__)
logger_format = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_format)
main_logger.addHandler(stream_handler)