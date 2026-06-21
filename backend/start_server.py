#!/usr/bin/env python
import os
import sys
import uvicorn

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
