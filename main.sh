#!/bin/bash

nohup python3 -u VAE.py > ./log/expand_val/VAE.log 2>&1
nohup python3 -u AE.py > ./log/expand_val/AE.log 2>&1
nohup python3 -u betaVAE.py > ./log/expand_val/betaVAE.log 2>&1
nohup python3 -u conv1dVAE.py > ./log/expand_val/conv1dVAE.log 2>&1
# nohup python3 -u stftVAE.py > ./log/stftVAE.log 2>&1