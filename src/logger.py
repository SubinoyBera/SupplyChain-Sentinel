import os
import sys
import logging
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path= os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH= os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    format="[ %(asctime)s ]: %(levelname)s - %(lineno)d %(module)s - %(message)s",
    level=logging.INFO,
    
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)