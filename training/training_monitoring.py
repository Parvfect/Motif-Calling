
import wandb
import random
import os
from model_config import ModelConfig


def get_api_key(running_on_hpc: bool = False):

    start_path = ""
    if running_on_hpc:
        start_path = os.environ['HOME']

    api_key_path = os.path.join(
        start_path, "wandb_api_key.txt")

    with open(api_key_path, 'r') as f:
        api_key = f.readline()
    return api_key.strip().replace('\n', '')

def wandb_login(running_on_hpc:bool = False):
    wandb.login(key=get_api_key(running_on_hpc=running_on_hpc))
    
def start_wandb_run(model_config: ModelConfig, project_name: str):
    wandb.init(
        project=project_name,
        config=model_config.__dict__()
    )