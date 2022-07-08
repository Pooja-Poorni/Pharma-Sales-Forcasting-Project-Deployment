# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:45:20 2022

@author: KanchanadeviM
"""

import requests

url = 'http://localhost:5000/predict for next 7 days_api'
r = requests.post(url,json={'SalesDate': "Date"})

print(r.json())

import os
os.chdir(r"C:\Users\KanchanadeviM\Document\Project code")