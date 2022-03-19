import sys, os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


def put(source):
    files = os.listdir(source)
    for f in files:
        file_path = source + f
        print(f'file_path: {file_path}')
    return files, file_path


files = put("output/")
# print(len(files))

# print(files[0])
x = 0
for f in files[0]:  
    path = files[0][x]
    x +=1
    img = ("output/" + path)

    # img = Image.open(img)

    # img.show() 


generate_res = 3
generate_square = 32 * generate_res 
image_channels = 3

# Previw image
peview_rows = 4
preview_cols = 7
preview_margin = 16

# Size vector to generate images from 
seed_size = 100

# Configuration
data_path = put("output/")
epcohs = 50
batch_size = 32
buffer_size = 600
# print(f"{generate_square}px images")





