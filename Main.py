# -*- coding:utf-8 -*-

###########################################################################
#                                                                         #
#          Program : Super Resolution Reconstruction                      #
#                                                                         #
###########################################################################

import os
import sys
import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import subprocess

from utils import imgs
from upsample import upsampling as sr
from utils import gen_lr as lrgen
from utils import check_train_data as dcheck
from train import training as tr
from utils import network_paras as paras
from test import create_test_data as ctest

class MainUi(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super().__init__()
        self.init_ui()
        # self.timer_status = False

    def init_ui(self):
        # set window title and size of init window
        self.setWindowTitle('super-resolution-reconstructer')  
        self.setFixedSize(1600, 950)
        
        # set icon
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(':gj.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        # framework of Main Window
        self.main_widget = QtWidgets.QWidget() # Instantiate a QWidget component as part of the main window
        self.main_widget.setObjectName('main_widget')
        self.main_layout = QtWidgets.QGridLayout() # create a layout
        self.main_widget.setLayout(self.main_layout) # Set the layout of the main part to a grid layout
        self.setStyleSheet('''
                                #datetime_label
                                {
                                    font-size:15px;
                                    font-weight:200;
                                    font-family: 'Consolas';
                                }
                                #at_icon
                                {
                                    background-image: url(:alogo.png);
                                    width : 200px;
                                    height : 100px;
                                }
                                #log_table
                                {
                                    color:red;
                                    font-size:18px;
                                    font-weight:200;
                                    font-family: 'Consolas';
                                }
                                QTabBar::tab
                                {
                                    min-width:40ex;
                                    min-height:10ex;
                                    font-family: '微软雅黑';
                                    font-size:15px;
                                    font-weight:90;
                                }
                            ''')

        ###########################################################################
        #                                                                         #
        #                           top widget                                    #
        #                                                                         #
        ###########################################################################
        
        # top widget : datatime_label, log_table, attention image
        self.widget_top = QtWidgets.QWidget() # create a widget
        self.widget_top_layout = QtWidgets.QGridLayout() # create a layout
        self.widget_top.setLayout(self.widget_top_layout) # Set the widget_top layout to grid layout

        # [1] datetime label to display system time
        self.datetime_label = QtWidgets.QLabel()
        self.datetime_label.setObjectName('datetime_label')
        self.datetime_label.setText('system time : {}'.format(datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))) 
        
        # [2] log label to display system logs after button pushing
        self.log_table = QtWidgets.QLabel()
        self.log_table.setObjectName('log_table')
        self.shutdown_lable = QtWidgets.QLabel()

        # [3] button to display attention image
        self.attention_img = QtWidgets.QPushButton()
        self.attention_img.setObjectName('at_icon')
        self.attention_img.clicked.connect(self.attention_image)

        # Add widgets to the upper layout
        self.widget_top_layout.addWidget(self.datetime_label, 0, 0, 1, 3)
        self.widget_top_layout.addWidget(self.log_table, 1, 0, 1, 3)
        self.widget_top_layout.addWidget(self.attention_img, 0, 4)
        
        ###########################################################################
        #                                                                         #
        #                       buttom widget                                     #
        #                                                                         #
        ###########################################################################

        # buttom widget
        self.widget_bottom = QtWidgets.QTabWidget()
        # create tabs
        self.upsampling_tab = QtWidgets.QTabWidget() # upsample tab
        self.data_check_tab = QtWidgets.QTabWidget() # data checking tab
        self.training_tab = QtWidgets.QTabWidget() # training_tab model tab
        self.display_net_tab = QtWidgets.QTabWidget() # display paras of network tab
        self.testing_model_tab = QtWidgets.QTabWidget() # testing_model_tab model tab
        self.help_tab = QtWidgets.QTabWidget() # help tab

        ###########################################################################
        #                                                                         #
        #                       TAB 1 ~ 7                                         #
        #                                                                         #
        ###########################################################################

        # [TAB_1] first tab : upsampling tab
        self.upsample_tab_layout = QtWidgets.QGridLayout() 
        self.upsampling_tab.setLayout(self.upsample_tab_layout)
        # raw_image_view, hr_image_view, loading_data_button, sr_button
        self.imageView = QLabel('Raw Resolution Image')
        self.imageView.setAlignment(Qt.AlignCenter)
    
        self.hr_image = QLabel('High Resolution Image')
        self.hr_image.setAlignment(Qt.AlignCenter)
    
        self.loadImage = QtWidgets.QPushButton('loadImage')
        self.loadImage.setObjectName('loadImage')
        self.loadImage.clicked.connect(self.load_images)

        self.sr_image = QtWidgets.QPushButton('sr_image')
        self.sr_image.setObjectName('sr_image')
        self.sr_image.clicked.connect(self.sr_images)
    
        self.loadImage.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.sr_image.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.hr_image.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'green';")
        self.imageView.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'green';")
        # append buttons and labels into tab layout
        self.upsample_tab_layout.addWidget(self.imageView, 0, 0)
        self.upsample_tab_layout.addWidget(self.hr_image, 0, 1)
        self.upsample_tab_layout.addWidget(self.loadImage, 1, 0)
        self.upsample_tab_layout.addWidget(self.sr_image, 1, 1)
        
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # [TAB_2] second tab : down-sampling tab
        self.downsampling_tab = QtWidgets.QTabWidget()
        self.downsampling_tab_layout = QtWidgets.QGridLayout()
        self.downsampling_tab.setLayout(self.downsampling_tab_layout)
        
        # raw images label and low resolution images label
        self.raw_images_label = QLabel('Here to Display Raw Resolution Image')
        self.raw_images_label.setAlignment(Qt.AlignCenter)
        self.lr_images_label = QLabel('Here to Display Low Resolution Image')
        self.lr_images_label.setAlignment(Qt.AlignCenter)

        # loading raw images button and loading low resolution imags button
        self.raw_images = QtWidgets.QPushButton('raw_images')
        self.raw_images.setObjectName('raw_images')
        self.raw_images.clicked.connect(self.raw_images_button_push)
        self.lr_images = QtWidgets.QPushButton('lr_images')
        self.lr_images.setObjectName('lr_images')
        self.lr_images.clicked.connect(self.lr_images_button_push)

        # start down sampling button
        self.start_dsample = QtWidgets.QPushButton('down-sample')
        self.start_dsample.setObjectName('down_sampling')
        self.start_dsample.clicked.connect(self.down_sampling)
        self.start_dsample.sizeHint()

        # set up progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setAlignment(Qt.AlignCenter)
        self.timer = False
        self.step = 0

        # set up style of labels - raw_images_label, lr_images_label, raw_images_button, lr_images_label, start_down_sampling_button
        self.raw_images_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'purple';")
        self.lr_images_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'purple';")
        self.raw_images.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.lr_images.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.start_dsample.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.downsampling_tab_layout.addWidget(self.start_dsample, 1, 0, 1, 2)
        self.downsampling_tab_layout.addWidget(self.pbar, 2, 0, 1, 2)
        self.downsampling_tab_layout.addWidget(self.raw_images_label, 3, 0, 2, 1)
        self.downsampling_tab_layout.addWidget(self.lr_images_label, 3, 1, 2, 1)
        self.downsampling_tab_layout.addWidget(self.raw_images, 4, 0)
        self.downsampling_tab_layout.addWidget(self.lr_images, 4, 1)        

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # [TAB_3] third tab : checking data tab
        self.data_check_tab = QtWidgets.QTabWidget()
        self.data_check_tab_layout = QtWidgets.QGridLayout()
        self.data_check_tab.setLayout(self.data_check_tab_layout)

        self.dataset_label = QLabel('Information of DataSet for Training.')
        self.dataset_label.setAlignment(Qt.AlignCenter)
        self.dataset_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 100 14pt \"Consolas\";"
                                        "color: 'purple';")

        self.dataset_info = QtWidgets.QPushButton('click to display information of dataset for training')
        self.dataset_info.setObjectName('dataset_info')
        self.dataset_info.clicked.connect(self.display_dataset_info)
        self.dataset_info.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")

        self.data_check_tab_layout.addWidget(self.dataset_info, 1, 0)
        self.data_check_tab_layout.addWidget(self.dataset_label, 2, 0, 6, 1)

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # [TAB_4] 4th tab : training model tab
        self.training_tab = QtWidgets.QTabWidget()
        self.training_tab_layout = QtWidgets.QGridLayout()
        self.training_tab.setLayout(self.training_tab_layout)

        self.training_button = QtWidgets.QPushButton('click to train SRGAN')
        self.training_button.setObjectName('training_button')
        self.training_button.clicked.connect(self.training_process)
        self.training_button.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        
        self.pretrain_label = QLabel('Pre-Training Phase.')
        self.pretrain_label.setAlignment(Qt.AlignCenter)
        self.pretrain_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 100 14pt \"Consolas\";"
                                        "color: 'purple';")
        self.train_label = QLabel('Training Phase.')
        self.train_label.setAlignment(Qt.AlignCenter)
        self.train_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 100 14pt \"Consolas\";"
                                        "color: 'purple';")

        # set up progress bar of pre-train and train
        self.pretrain_bar = QProgressBar(self)
        self.pretrain_bar.setAlignment(Qt.AlignCenter)

        self.train_bar = QProgressBar(self)
        self.train_bar.setAlignment(Qt.AlignCenter)

        self.train_image = QLabel('')
        self.train_image.setAlignment(Qt.AlignCenter)
        self.train_image.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "background-image: url(./backgroud/train.png);"
                                        "font: 100 14pt \"Consolas\";"
                                        "color: 'purple';")
        
        self.training_tab_layout.addWidget(self.training_button, 1, 0, 1, 2)
        self.training_tab_layout.addWidget(self.pretrain_label, 2, 0)
        self.training_tab_layout.addWidget(self.train_label, 2, 1)
        self.training_tab_layout.addWidget(self.pretrain_bar, 3, 0)
        self.training_tab_layout.addWidget(self.train_bar, 3, 1)
        self.training_tab_layout.addWidget(self.train_image, 4, 0, 4, 2)

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # [TAB_5] 5th tab : display network paras tab
        self.display_net_tab = QtWidgets.QTabWidget()
        self.display_net_tab_layout = QtWidgets.QGridLayout()
        self.display_net_tab.setLayout(self.display_net_tab_layout)

        self.net_button = QtWidgets.QPushButton('click to display network paras')
        self.net_button.setObjectName('net_button')
        self.net_button.clicked.connect(self.display_net)
        self.net_button.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        
        self.text_browser = QTextBrowser(self)
        self.text_browser.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 100 14pt \"Consolas\";"
                                        "color: 'black';")

        self.display_net_tab_layout.addWidget(self.net_button, 1, 0)
        self.display_net_tab_layout.addWidget(self.text_browser, 2, 0, 5, 1)

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # [TAB_6] 6th tab : testing model tab
        self.testing_model_tab = QtWidgets.QTabWidget()
        self.testing_model_tab_layout = QtWidgets.QGridLayout()
        self.testing_model_tab.setLayout(self.testing_model_tab_layout)

        # button : create test data
        self.create_testdata = QtWidgets.QPushButton('click to create data for testing.')
        self.create_testdata.setObjectName('create_testdata')
        self.create_testdata.clicked.connect(self.create_test_data)
        # self.create_testdata.sizeHint()
        self.create_testdata.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")

        # progress bar : record progress for creating testing data
        self.test_bar = QProgressBar(self)
        self.test_bar.setAlignment(Qt.AlignCenter)
        self.test_timer = False
        self.test_step = 0
        
        # 3 labels : low_res, high_fake, high_real
        self.low_res_label = QLabel('Here to Display low resolution images.')
        self.low_res_label.setAlignment(Qt.AlignCenter)
        self.high_real_label = QLabel('Here to Display High Resolution Real images.')
        self.high_real_label.setAlignment(Qt.AlignCenter)
        self.high_fake_label = QLabel('Here to Display High Resolution Fake images.')
        self.high_fake_label.setAlignment(Qt.AlignCenter)

        self.low_res_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'purple';")
        self.high_real_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'purple';")
        self.high_fake_label.setStyleSheet(
                                        "background-color: 'lightgrey';"
                                        "font: 70 12pt \"Consolas\";"
                                        "color: 'purple';")

        # 3 buttons : loading low_res, high_res_real, high_res_fake
        self.low_res_button = QtWidgets.QPushButton('low_res')
        self.low_res_button.setObjectName('low_res_button')
        self.low_res_button.clicked.connect(self.low_res_button_push)
        
        self.high_res_real_button = QtWidgets.QPushButton('high_res_real')
        self.high_res_real_button.setObjectName('high_res_real_button')
        self.high_res_real_button.clicked.connect(self.high_res_real_button_push)

        self.high_res_fake_button = QtWidgets.QPushButton('high_res_fake')
        self.high_res_fake_button.setObjectName('high_res_fake_button')
        self.high_res_fake_button.clicked.connect(self.high_res_fake_button_push)

        self.low_res_button.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.high_res_real_button.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
        self.high_res_fake_button.setStyleSheet(
                                        "background-color: 'gold';"
                                        "border-color: 'purple';"
                                        "font: 75 12pt \"Consolas\";"
                                        "color: 'black';")
     
        self.testing_model_tab_layout.addWidget(self.create_testdata, 1, 0, 1, 3)
        self.testing_model_tab_layout.addWidget(self.test_bar, 2, 0, 1, 3)
        
        self.testing_model_tab_layout.addWidget(self.low_res_label, 3, 0, 2, 1)
        self.testing_model_tab_layout.addWidget(self.high_real_label, 3, 1, 2, 1)
        self.testing_model_tab_layout.addWidget(self.high_fake_label, 3, 2, 2, 1)
        
        self.testing_model_tab_layout.addWidget(self.low_res_button, 4, 0)
        self.testing_model_tab_layout.addWidget(self.high_res_real_button, 4, 1)
        self.testing_model_tab_layout.addWidget(self.high_res_fake_button, 4, 2)

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

        # [TAB_7] 7th tab : help tab
        self.help_tab = QtWidgets.QTabWidget()
        self.help_tab_layout = QtWidgets.QGridLayout()
        self.help_tab.setLayout(self.help_tab_layout)

        self.help = QLabel('\t[帮助文本] 这是一个关于程序的描述')
        self.help.setStyleSheet(
                                        "background-color: 'white';"
                                        "font: 70 12pt \"微软雅黑\";"
                                        "color: 'black';")
        self.emoij = QLabel('')
        self.emoij.setStyleSheet(
                                        "background-color: 'white';"
                                        "background-image: url(./backgroud/pytorch.png);"
                                        "font: 70 12pt \"微软雅黑\";"
                                        "color: 'black';")
        
        text = '''\n\n超分辨率技术(SR，Super-Resolution)是指从观测到的低分辨率\n
                  图像重建出相应的高分辨率图像,在监控设备、卫星图像和医学影像\n
                  等领域都有重要的应用价值.传感器制造技术对图像分辨率存在限制,\n
                  一种具有前景的解决方法就是采用信号处理技术通过多帧低分辨率\n
                  (LR，Low Resolution)观察图像来获得一帧高分辨率图像或者高分辨率\n
                  图像序列.采用信号处理方法的主要优点是成本低,并且现有的低分辨率成\n
                  像系统仍然可以被利用.SR可分为两类:从多张低分辨率图像重建出高分辨\n
                  率图像和从单张低分辨率图像重建出高分辨率图像.基于深度学习的SR,主\n
                  要是基于单张低分辨率的重建方法,即\n
                  Single Image Super-Resolution (SISR).'''
        
        self.help.setText(text)
        self.help.setAlignment(Qt.AlignLeft)
        self.help_tab_layout.addWidget(self.help, 1, 0)
        self.help_tab_layout.addWidget(self.emoij, 1, 1)

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
        
        # append tabs into widget
        self.widget_bottom.addTab(self.upsampling_tab, '图像超分辨率重建')
        self.widget_bottom.addTab(self.downsampling_tab, '图像下采样')
        self.widget_bottom.addTab(self.data_check_tab, '训练数据')
        self.widget_bottom.addTab(self.training_tab, '模型训练')
        self.widget_bottom.addTab(self.display_net_tab, '网络参数')
        self.widget_bottom.addTab(self.testing_model_tab, '模型测试')
        self.widget_bottom.addTab(self.help_tab, '帮助')

        # append widgets to main layout
        self.main_layout.addWidget(self.widget_top, 0, 0, 1, 1)
        self.main_layout.addWidget(self.widget_bottom, 1, 0, 4, 1)
        
        # Set the UI core component to main_widget
        self.setCentralWidget(self.main_widget) 

        # Real time timer
        self.datetime = QtCore.QTimer()  # Instantiate a timer
        self.datetime.setInterval(1000)  # Set the timer interval 1 second
        self.datetime.start()  # Start timer
        # Timer is connected to the slot function to update the UI interface time
        self.datetime.timeout.connect(self.show_datetime_slots)  
    
    ###########################################################################
    #                                                                         #
    #                           slot functions                                #
    #                                                                         #
    ###########################################################################

    def attention_image(self):
        self.log_table.setText('Program : super resolution reconstruction implemented by PyTorch.\nAuthor : Zhong Jia-Rong\nDate : 2019-9-3')
    
    # loading images to window
    def load_images(self):
        self.filename = QFileDialog.getOpenFileName(self, 'OpenFile', '.', 'Image Files(*.jpg *.jpeg *.png)')[0]
        if len(self.filename):
            # loading raw images
            self.image = QImage(self.filename)
            self.imageView.setPixmap(QPixmap.fromImage(self.image))
            self.imageView.setScaledContents(True)

    def raw_images_button_push(self):
        self.filename = QFileDialog.getOpenFileName(self, 'OpenFile', '.', 'Image Files(*.jpg *.jpeg *.png)')[0]
        if len(self.filename):
            # loading raw images
            self.image = QImage(self.filename)
            self.raw_images_label.setPixmap(QPixmap.fromImage(self.image))
            self.raw_images_label.setScaledContents(True)

    def lr_images_button_push(self):
        self.filename = QFileDialog.getOpenFileName(self, 'OpenFile', '.', 'Image Files(*.jpg *.jpeg *.png)')[0]
        if len(self.filename):
            # loading raw images
            self.image = QImage(self.filename)
            self.lr_images_label.setPixmap(QPixmap.fromImage(self.image))
            self.lr_images_label.setScaledContents(True)
    
    # super resolution reconstruction and loading hr images to window
    def sr_images(self):  
        if len(self.filename):
            # create sr images
            log = sr(self.filename, 'hr_image.png', 4)
            # reset self.filename to hr_image.png
            parts = self.filename.split('/')
            parent = parts[0:len(parts)-2]
            parent.append('output')
            parent.append('result')
            parent.append('hr_image.png')
            self.filename = '/'.join(parent)
            # loading sr images from filepath
            self.image = QImage(self.filename)
            self.hr_image.setPixmap(QPixmap.fromImage(self.image))
            self.hr_image.setScaledContents(True)
            # display sr log
            self.log_table.setText(log)

    # display system time
    def show_datetime_slots(self):
        self.datetime_label.setText('system time : {}'.format(datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')))

    # start down sampling
    def down_sampling(self):
        if self.timer == True:
            # self.timer.stop()
            self.timer = False
            self.start_dsample.setText('start down sampling')
        else:
            # self.timer.start(100, self)
            self.timer = True
            self.start_dsample.setText('stop down sampling')
        
            # images down-sampling
            input_dir = os.getcwd() + r'\down_sampling\input_file'
            output_dir = os.getcwd() + r'\down_sampling\output_file'
            lrgen.lower_resolution_images(input_dir, output_dir, self.step, self.pbar)
            
            self.log_table.setText('InputPath : {} 请将需要下采样的图像数据置于此文件夹内.\nOutputPath : {} 下采样结束后请在此文件夹下获取低分辨率图像.'.format(input_dir, output_dir))
            self.start_dsample.setText('complete')

    def display_dataset_info(self):
        data_dir = os.getcwd() + r'\traindata\Set5(228-256)\\'
        log = dcheck.images_check(data_dir)
        self.dataset_label.setText(log)

    def training_process(self):
        tr(self.pretrain_bar, self.train_bar)

    def display_net(self):
        pfile = 'network.py'
        lfile = os.getcwd() + '\\backgroud\\log.txt'
        res = subprocess.Popen(
                                'python ' + pfile + ' > ' + lfile,
                                stdout = subprocess.PIPE,
                                shell = True
                              ).communicate()

        model_paras = ''
        with open(lfile, 'r') as file:
            for i in file.readlines():
                model_paras += i
        
        self.text_browser.append(model_paras)
        self.text_browser.moveCursor(self.text_browser.textCursor().End)

    def create_test_data(self):
        if self.test_timer == True:
            self.test_timer = False
            self.create_testdata.setText('start creating test data')
        else: 
            self.test_timer = True
            self.create_testdata.setText('stop creating test data')
            path = os.getcwd() + r'\images\\'
            lr_path, hr_real_path, hr_fake_path = ctest(path, self.test_step, self.test_bar)
            self.log_table.setText('low_res 文件位置 : {}\nhigh_res_real 文件位置 : {}\nhigh_res_fake 文件位置 : {}'.format(lr_path, hr_real_path, hr_fake_path))
            self.start_dsample.setText('complete')

    def low_res_button_push(self):
        self.filename = QFileDialog.getOpenFileName(self, 'OpenFile', '.', 'Image Files(*.jpg *.jpeg *.png)')[0]
        if len(self.filename):
            self.image = QImage(self.filename)
            self.low_res_label.setPixmap(QPixmap.fromImage(self.image))
            self.low_res_label.setScaledContents(True)

    def high_res_real_button_push(self):
        self.filename = QFileDialog.getOpenFileName(self, 'OpenFile', '.', 'Image Files(*.jpg *.jpeg *.png)')[0]
        if len(self.filename):
            self.image = QImage(self.filename)
            self.high_real_label.setPixmap(QPixmap.fromImage(self.image))
            self.high_real_label.setScaledContents(True)

    def high_res_fake_button_push(self):
        self.filename = QFileDialog.getOpenFileName(self, 'OpenFile', '.', 'Image Files(*.jpg *.jpeg *.png)')[0]
        if len(self.filename):
            self.image = QImage(self.filename)
            self.high_fake_label.setPixmap(QPixmap.fromImage(self.image))
            self.high_fake_label.setScaledContents(True)

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
