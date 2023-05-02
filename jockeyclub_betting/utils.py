from functools import wraps
import logging
import yaml
import os
from pathlib import Path
PROJECT_PATH = Path(os.getcwd())
FORMAT = '%(asctime)s %(levelname)s: %(message)s'
LEVEL = logging.DEBUG
logging.basicConfig(format=FORMAT,filename='./logs/executionLog.log',filemode='w',level=LEVEL)
#logging.basicConfig(format=FORMAT,level=LEVEL)

def logged(func):
    @wraps(func)
    def with_logging(*args,**kwargs):
        logging.debug(func.__qualname__+" was called")
        return func(*args,**kwargs)
    return with_logging

def read_credentials(credential_path):
    logging.info('Parsing credentials from yaml files...')
    try:
        with open(credential_path, 'r') as stream:
            creds = yaml.safe_load(stream)
            return creds
    except Exception as e:
        logging.exception(f'{e}',exc_info=True)