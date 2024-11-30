import os
import sys
import yaml
from src.logger import logging
from src.exception import CustomException
from ensure import ensure_annoations
from pathlib import Path

@ensure_annoations
def read_yaml(path_to_yaml: Path):
    try:
        with open(path_to_yaml) as yaml_file:
            content= yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully.")
            
            return content
        
    except Exception as e:
        raise CustomException(e,sys)