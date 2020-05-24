import os
import cv2
import random
import numpy as np
import tensorflow as tf
import model.utils as utils
import model.config as cfg

class Dataset(object):
    '''
    Data process
    '''
    def __init__(self, dataset_type):
        self.anno_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_name(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotation = self.load_annotaions(dataset_type)
        self.num_sample = len(self.annotation)
        self.num_batch = int(np.ceil(self.num_sample) / self.batch_size)
        self.batch_count = 0

    def load_annotaions(self, dataset_type)->list:
        '''
        read annotations
        :param dataset_type: this can change
        :return: list type
        '''
        with open(self.anno_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            np.random.shuffle(annotations)
            return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes) #随机选取训练照片尺寸
            self.train_output_sizes = self.train_input_size // self.strides #

            batch_image = np.zeros((self.batch_size, self.train_input_sizes, self.train_input_size, 3))

            batch_label_sbbox = np.zeros((self.batch_size, self.train_input_sizes[0], self.train_input_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_input_sizes[1], self.train_input_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_input_sizes[2], self.train_input_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            #调用for Dataset可迭代对象
            if self.batch_count < self.num_batch: 
                #赋值操作
                while num < self.batch_size:
                    index = self.batch_count
