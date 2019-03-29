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

save_dir = os.path.abspath('trial1610r')

csvfile = open("validation.csv", "w")

def join(dirpath, filename):
    return os.path.join(dirpath, filename)

def main(data_set):
    miny=10
    maxy=20
    inputs = DataSetLoader.load_validation_images(data_set)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    model_name = 'test.ckpt'
    model_path = join(save_dir, model_name)
    saver.restore(sess, model_path)
    writer = csv.writer(csvfile)
    numFiles = len(inputs)
    print("Total number of images collected: %d" %numFiles)
    for i in range(0,numFiles-1):
        data=[]
        image=inputs[i]
        img = cv2.resize(image, (200, 66))
        img = img / 255.
        curtime = time.time()
        steer = model.y.eval(feed_dict={model.x: [img]})[0][0]
        steering=(float(steer)*(maxy-miny))+miny
        steering=round(steering, 2)
        pred_time = time.time() - curtime
        data.append(pred_time)
        data.append(steering)
        writer.writerow(data)
        
if __name__ == '__main__':
    data_set = "path/to/data.csv"
    main(data_set)
