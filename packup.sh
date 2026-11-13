#!/bin/bash
rm -rf ./myrobot.tar.gz
tar -zcvf myrobot.tar.gz task train play myrobot config __init__.py robot_cfg.py
