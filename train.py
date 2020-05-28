import os
import time
import shutil
import numpy as np
import tensorflow as tf
import model.utils as utils
from tqdm import tqdm
from model.dataset import Dataset
from model.yolov3 import YOLOV3
from model.config import cfg

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_name(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.per_epch_num = len(self.trainset)
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(allow_soft_placement = True)) #GPU 自动调用

        with tf.name_scope('define_input'):
            self.input_data = tf.compat.v1.placeholder(dtype = tf.float32, name='input_data')
            self.label_sbbox = tf.compat.v1.placeholder(dtype= tf.float32, name = 'label_sbbox')
            self.label_mbbox = tf.compat.v1.placeholder(dtype = tf.float32, name = 'label_mbbox')
            self.label_lbbox = tf.compat.v1.placeholder(dtype = tf.float32, name= 'label_lbbox')
            self.true_mbbox = tf.compat.v1.placeholder(dtype = tf.float32, name = 'true_mbbox')
            self.true_sbbox = tf.compat.v1.placeholder(dtype = tf.float32, name = 'true_sbbox')
            self.true_lbbox = tf.compat.v1.placeholder(dtype = tf.float32, name = 'true_lbbox')
            self.trainable =tf.compat.v1.placeholder(dtype = tf.bool, name = 'training')

        with tf.name_scope('define_loss'):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.compat.v1.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_mbbox, self.label_lbbox, self.label_sbbox, self.true_lbbox, self.true_sbbox, self.true_mbbox
            )
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype = tf.float64, trainable = False, name = 'global_step')
            # why set warmup_setps
            warmup_setps = tf.constant(self.warmup_periods * self.per_epch_num)
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.per_epch_num,
                                      dtype= tf.float64, name='train_steps')
            # training learn rate how to change
            self.learn_rate = tf.cond(
                pred = self.global_step < warmup_setps, #预热 ，周期性变化在最大最小学习率之间
                true_fn=lambda : self.global_step / warmup_setps * self.learn_rate_init,
                false_fn = lambda : self.learn_rate_end + 0.5*(self.learn_rate_init - self.learn_rate_end)*
                                    (1 + tf.cos(
                                        (self.global_step - warmup_setps) / (train_steps - warmup_setps) * np.pi)
                                    )
            )
            global_setp_update = tf.compat.v1.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.compat.v1.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_trainable_var_list = []
            for var in tf.compat.v1.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_trainable_var_list.append(var)
            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                               var_list = self.first_trainable_var_list)
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                #保存训练之前完成的一些操作 优化器，步数的变化
                with tf.control_dependencies([first_stage_optimizer, global_setp_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.compat.v1.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                                var_list = second_stage_trainable_var_list)

            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                #保存训练之前完成的一些操作 优化器，步数的变化
                with tf.control_dependencies([second_stage_optimizer, global_setp_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_sever'):
            self.loader = tf.compat.v1.train.Saver(self.net_var) #保存全局变量
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir) #递归删除文件夹中内容
            os.makedirs(logdir)
            self.write_op = tf.compat.v1.summary.merge_all() #可以将所有summary全部保存到磁盘，以便tensorboard显示
            self.summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=self.sess.graph) #保存图结构

    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight) #初始化权重设置
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0 #训练epoch数量减少
        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs): #遍历每个epoch
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbbox: train_data[4],
                                                self.true_mbbox: train_data[5],
                                                self.true_lbbox: train_data[6],
                                                self.trainable:    True,
                })
                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss) #每个batch loss

            for test_data in self.testset:
                test_step_loss = self.sess.run( self.loss, feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbbox: test_data[4],
                                                self.true_mbbox: test_data[5],
                                                self.true_lbbox: test_data[6],
                                                self.trainable:    False,
                })
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)

if __name__ == '__main__': YoloTrain().train()
