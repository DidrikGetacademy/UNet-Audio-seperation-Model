import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from Training.Externals.utils import Return_root_dir   
root_dir = Return_root_dir()

import logging
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(file_handler)
    return logger


