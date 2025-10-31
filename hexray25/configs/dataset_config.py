import socket
from pathlib import Path
from typing import Dict

DSNAME: Dict[str, str] = {
    "ds1": "phase3_repeat_31may23.json",
    "ds2": "phase3a_30nov23_coco1.json",
    "ds3": "phase6_27apr24.json",
    "ds4": "phase6_23july24.json",
}

# DEFAULT_ROOT = Path("/home/ubuntu/foundations/vjt-data")
# DEFAULT_STORE_ROOT = Path("/home/ubuntu/foundations/vjt-data-2/") 

# Want to be able to share this file over different computers
hostname = socket.gethostname()
if hostname == "VJT-DLL3L84":
    DEFAULT_ROOT = Path("/home/ageorge/foundations/vjt-data")
    WANDB_STUB = "ag4090"
    REPO_BASE = Path("/home/ageorge/src/hexray25")
    #DEFAULT_STORE_ROOT = Path("/home/ageorge/foundations/ds") 
    DEFAULT_STORE_ROOT = Path("/home/ageorge/src/hexray25/hexray25_train/ds/") 
else:
    DEFAULT_ROOT = Path("/home/ubuntu/foundations/vjt-data/")
    DEFAULT_STORE_ROOT = Path("/home/ubuntu/prakash_v6/hexray25/hexray25_train/ds/") 
    WANDB_STUB = hostname
    REPO_BASE = Path("/home/ubuntu/prakash_v6/hexray25")
