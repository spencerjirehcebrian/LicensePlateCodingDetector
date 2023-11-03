import subprocess
import urllib.request
import os
import tkinter as tk
from tkinter import ttk
import cv2
from IPython import get_ipython

def run_yolov7_detection():
    # Start an IPython session
    ipython = get_ipython()

    # Now, you can run Jupyter-like commands
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

    # You can also run code cells
    ipython.run_cell('print("Hello, Jupyter!")')
    
def run(file_path):
    print('Converting to frames...')
    #convert_to_frames(file_path)
    print('Running Detection on Images...')
    run_yolov7_detection()
    print('Converting to Video...')
    #convert_to_video('output_video.mp4')
    print('Detection Complete')
 
run('random')    

    
