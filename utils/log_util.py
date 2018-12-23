# -*- coding: utf-8 -*-

'''
# Author  : Ming

# Time    : 2018/10/22 0022 上午 10:41

'''

import logging.config
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

log_path = os.environ.get('ANTISPAM_LOG_PATH')

if log_path:
    log_file=log_path+"/logging.conf"
else:     
    log_file=os.path.dirname(__file__)+"/logging.conf"

print("load log file:"+log_file)

logging.config.fileConfig(log_file)

log = logging.getLogger("root")  

def log_params(FLAGS):
    for k, v in FLAGS.__dict__.items():
        log.info(k + ":" + str(v))