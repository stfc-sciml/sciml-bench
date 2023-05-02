import os
from os import listdir
from PIL import Image

base_dir = "/home/lkt24131/sciml_bench/datasets/em_graphene_sim/inference-raw/"


for filename in listdir(base_dir):
    if filename.endswith('.tif'):
        try:
            img = Image.open(base_dir+"/"+filename) # open the image file
            img.verify() # verify that it is, in fact an image
            print(f'Image file {filename} is OK')
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)