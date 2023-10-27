import os
import glob
from distutils.util import strtobool

GPU = strtobool(os.getenv('GPU', 'true'))

# API_SERVER_IP = os.getenv('API_SERVER_IP', 'https://cartoru.net/')
# API_SERVER_IP = os.getenv('API_SERVER_IP', 'http://3.112.39.147')
# API_UNPROCESSED_PATH = os.getenv('API_UNPROCESSED_PATH', '/api/ai/exterior')
API_SERVER_IP = os.getenv('API_SERVER_IP', 'http://0.0.0.0:5004')
API_UNPROCESSED_PATH = os.getenv('API_UNPROCESSED_PATH', '/')
API_PROCESSED_PATH = os.getenv('API_PROCESSED_PATH', '/api/ai/exterior')
API_KEY = os.getenv('API_KEY', 'CHOWAGIKEN')

#For InfiniteLoop Production Server
      # - API_SERVER_IP=https://cartoru.net/
      # - API_UNPROCESSED_PATH=/api/ai/exterior
      # - API_PROCESSED_PATH=/api/ai/exterior
