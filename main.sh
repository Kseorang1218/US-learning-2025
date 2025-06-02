#!/bin/bash

# nohup python3 -u main.py > ./log/main.log 2>&1
# nohup python3 -u mainAE.py > ./log/mainAE.log 2>&1
# nohup python3 -u mainbetaVAE.py > ./log/mainbetaVAE.log 2>&1
# nohup python3 -u mainconv1dVAE.py > ./log/mainconv1dVAE.log 2>&1
nohup python3 -u mainstftVAE.py > ./log/mainstftVAE.log 2>&1