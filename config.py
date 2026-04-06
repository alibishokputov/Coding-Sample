import pandas as pd
import sys
import os
import json
import yaml
from pathlib import Path
from types import SimpleNamespace

def read_yaml(path: str):
    root_dir = Path(__file__).resolve().parent.parent 
    config_path = root_dir / 'references' / path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_configurations():
    cfg = SimpleNamespace(**read_yaml('configs.yaml'))
    cfg.supporting_data_path = os.path.join(cfg.drive_path, 
                                            cfg.supporting_data_drive_path)
    cfg.processed_data_drive_path = os.path.join(cfg.drive_path, 
                                            cfg.processed_data_drive_path)
    return cfg

