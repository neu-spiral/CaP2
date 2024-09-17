import logging
import os, sys

import torch.types

# Logger initialization
logger = logging.getLogger(__name__)

# Set the overall logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# Create handlers: one for file and one for console
file_handler = logging.FileHandler('logfile.log')
console_handler = logging.StreamHandler(sys.stdout)

# Set the logging level for each handler (DEBUG, INFO, WARNING, ERROR, CRITICAL)
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.