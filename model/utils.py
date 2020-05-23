# import cv2
# import random
# import colorsys
import numpy as np
# import tensorflow as tf
# import config as cfg

def read_class_name(filename):
    names = {}
    with open(filename, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return name

def get_anchors(anchor_file):
    '''loads the anchors from a file'''
    with open(anchor_file) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype = np.float32)
    a = anchors.reshape(3, 3, 2)
    print(a)
    return anchors.reshape(3,3,2)

if __name__ == '__main__':
    #filename = 'C:/Users/endingback/Desktop/tensorflow_yolov3/data/class/voc.names'
    #filename = './data/classes/coco.names'
    filename = 'C:/Users/endingback/Desktop/tensorflow_yolov3/data/anchors/coco_anchors.txt'
    get_anchors(filename)