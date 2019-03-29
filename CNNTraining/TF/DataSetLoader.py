#DataSetLoader.py
#Authored: Matthew Burruss
#Last Edited: 3/28/2019
#Description: Preprocessing images for multiple Trials of data

import random
import os
import sys
import cv2
import csv
import glob
import numpy as np

def load_training_images(pathToFile):
    # initialize loop variables
    inputs = []
    Y = []
    if not os.path.isfile(pathToFile):
        print("File not found.")
        return
    # read images in ProcessData.csv and append to input
    with open(pathToFile, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        list = []
        print("opened dataset")
        rownum = 1
        img_num = -1
        images_processed = 0
        for row in reader:
            if (images_processed == img_num):
                print("Read: %d images" %images_processed)
                break
            if (rownum == 1):
                img_num = int(row[0])
                rownum = rownum + 1
                continue
            else:
                list.append(row)
            # every other row: get the images (each image is spread 13200x3 in excel)
            if ((rownum-1) %3 == 0):
                img = np.asarray(list,dtype=np.uint8)
                img = np.resize(img,(66,200,3))
                img = img / 255.
                inputs.append(img)
                list = []
                images_processed = images_processed + 1
            rownum = rownum + 1
    return inputs

def read_training_output_data(pathToFile):
    # initialize loop variables
    Y=[]
    if not os.path.isfile(pathToFile):
        print("File not found.")
        return
    # read images in ProcessData.csv and append to input
    with open(pathToFile, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        list = []
        print("opened dataset")
        rownum = 1
        for row in reader:
            if (rownum == 1):
                img_num = int(row[0])
                rownum = rownum + 1
                continue
            else:
                list.append(row)
            """
            # timestamps
            if (rownum == img_num*3+2):
                print(list)
                rownum = rownum + 1
                continue
            # acc duty cycles
            if (rownum == img_num*3+3):
                print(list)
                rownum = rownum + 1
                continue
            """
            # steering duty cycles
            if (rownum == img_num*3+4):
                for steering in row:
                    output = []
                    y=(float(steering)-10.0)/(20.0-10.0)
                    output.append(y)
                    Y.append(output)
            rownum = rownum + 1
    return Y

def load_validation_images(pathToFile):
    return load_training_images(pathToFile)

def read_validation_output_data(pathToFile):
    return read_training_output_data(pathToFile)

def load_images_and_outputs_batch(pathToFile,batch_size):
    input = load_training_images(pathToFile)
    output = read_training_output_data(pathToFile)
    n = len(input)
    value = random.sample(range(0, n), batch_size)
    batch_images = []
    batch_outputs = []
    for i in value:
        batch_images.append(input[i])
        batch_outputs.append(output[i])
    return batch_images,batch_outputs

"""
if __name__ == '__main__':
    input = load_training_images(pathToFile="path/to/some/datafile/Data20190329003618207.csv")
    load_validation_images_batch
    output = read_validation_output_data(pathToFile="path/to/some/datafile/Data20190329003618207.csv)
    #print(output)
    #load_validation_images()
    #read_validation_output_data()
"""