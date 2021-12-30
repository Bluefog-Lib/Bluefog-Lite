import logging
import os

global_rank = os.getenv('BFL_WORLD_RANK')
logger = logging.getLogger("bluefog")
logger.setLevel(logging.WARNING)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(f"R{global_rank}: %(asctime)-15s %(levelname)s  %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)