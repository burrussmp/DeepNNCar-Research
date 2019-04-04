#Test.py
#Authored: Shreyas Ramakrishna and Matthew P. Burruss
#Last Edited: 3/28/2019
#Description: Script to validate the trained CNN model.

import cv2
import math
import numpy as np
import time
import os
import model
import DataSetLoader
import tensorflow as tf
import glob
import csv

save_dir = os.path.abspath('model_500')

csvfile = open("validation.csv", "w")

def join(dirpath, filename):
    return os.path.join(dirpath, filename)

def main(data_set):
    miny=10
    maxy=20
    inputs = DataSetLoader.load_validation_images(data_set)
    outputs = DataSetLoader.read_validation_output_data(data_set)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    model_name = 'test.ckpt'
    model_path = join(save_dir, model_name)
    saver.restore(sess, model_path)
    writer = csv.writer(csvfile)
    numFiles = len(inputs)
    print("Total number of images collected: %d" %numFiles)
    mse = 0.0
    for i in range(0,numFiles-1):
        data=[]
        img=inputs[i]
        curtime = time.time()
        steer = model.y.eval(feed_dict={model.x: [img]})[0][0]
        steering=(float(steer)*(maxy-miny))+miny
        steering=round(steering, 2)
        pred_time = time.time() - curtime
        data.append(pred_time)
        data.append(steering)
        out = outputs[i][0]
        actual = out*10.0+10
        actual = round(actual,2)
        mse = mse + (steering-actual)**2
        data.append(actual)
        writer.writerow(data)
    mse = mse / float(numFiles)
    print('Mean-squared error: %0.2f'%mse)
if __name__ == '__main__':
    data_set = '/home/scope/Burruss/DeepNNCar/DeepNNCar-Research/CNNTraining/TF/Data20190401035027367.csv'
    main(data_set)
