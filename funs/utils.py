# utils.py

import yaml
from box import Box

import numpy as np
import random

import torch

import argparse

def load_yaml(config_path: str) -> Box:
    """
    YAML 파일을 load하는 함수

    Parameters
    ----------
    config_path : str
        YAML 파일 경로

    Returns
    -------
    Box 
        Box 개체
    """
    with open(config_path) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)

    return config

def set_seed(seed: int):
    """
    랜덤 시드 고정 함수.

    Parameters
    ----------
    seed : int
        고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="KSNVE challenge")
    
    # latent space 인자
    parser.add_argument(
        "--latent_size_list", 
        type=int, 
        nargs="+", 
        default=[4096, 2048, 1024, 512],
        help="latent space 입력 (예: --latent_size_list 4096 2048 1024 512, 기본값: [4096, 2048, 1024, 512])"
    )
        
    parser.add_argument(
        "--latent_size",
        type=int,
        default=512,
        help="단일 latent space 크기 (예: --latent_size 512, 기본값: 512)"
    )
        
    args = parser.parse_args()

    return args
