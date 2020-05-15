#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  14 15:10:08 2020

@author: Deechean
"""
import os
import threading
import train_log
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def weight_graph_1(datalist):
    plt.rcParams['figure.dpi'] = 100     
    
    datalist = np.array(datalist).transpose(4,0,1,2,3)
    datalist = datalist.reshape([datalist.shape[0],datalist.shape[1],datalist.shape[2], datalist.shape[3]])
    
    fig, ax = plt.subplots(datalist.shape[0],figsize=(6,12))
    
    i = 0      
    for lay in datalist:
        j = 0
        for data in lay:
            y = np.array(data)
            x = np.linspace(0.0, y.shape[0]*y.shape[1],y.shape[0]*y.shape[1])             
            ax[i].plot(x, y.reshape([y.shape[0]*y.shape[1]]).tolist(), label='step_'+str(j))
            ax[i].set_title('Layer' + str(i))
            ax[i].grid(True)
            ax[i].legend()
            j += 1
        i += 1       
plt.show()    

def weight_graph_2(datalist):
    plt.rcParams['figure.dpi'] = 100     
    
    datalist = np.array(datalist).transpose(4,0,1,2,3)
    datalist = datalist.reshape([datalist.shape[0],datalist.shape[1],datalist.shape[2], datalist.shape[3]])
        
    i = 0   
    _x = np.arange(datalist.shape[2])  
    _y = np.arange(datalist.shape[3])  
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    for lay in datalist:
        start_value = lay[0]
        end_value = lay[len(lay)-1]
        change_value = end_value - start_value
        print('-----------Layer %d-----------'%i)
        print('change_value:')
        print(change_value)
        top = []
        for j in range(len(x)):
            top.append(change_value[x[j],y[j]])
        print(x)
        print(y)
        print(top)
        width = depth = 1
        bottom = np.zeros_like(top)
        fig = plt.figure(figsize=(4,4))
        ax = Axes3D(fig)
        ax.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax.set_title('Layer' + str(i))
        i += 1
    plt.show()
    
    
def bia_graph(datalist):
    plt.rcParams['figure.dpi'] = 100 
    fig_data=[]
    i = 0
    fig, ax = plt.subplots()
    for data in datalist:
        d = np.array(data)
        length = 1
        for j in range(len(d.shape)):
            length *= d.shape[j]        
        d = d.reshape(length).tolist()
        x = np.linspace(0.0, len(d), len(d)) 
        #print(x)
        #print(d)
        ax.plot(x, d, label='step_'+str(i))
        i += 1
        
    ax.set_title('Variable Graph')
    fig.tight_layout()
    plt.show()
    
def image_show_old(datalist):
    images_per_row = 6
    plt.rcParams['figure.dpi'] = 100     
    
    datalist = np.array(datalist).transpose([0,1,4,2,3])
    i = 0
    for image_batch in datalist:
        print('mini batch %d:'%i)
        j = 0
        for image in image_batch:
            k = 0
            fig, ax = plt.subplots(nrows=int((image.shape[0]-1)/images_per_row + 1),ncols=images_per_row,
                                   figsize=(2*images_per_row, 2*int((image.shape[0]-1)/images_per_row + 1)))            
            
            print('batch item '+str(j))            
            for layer in image:
                if int((image.shape[0]-1)/images_per_row + 1) == 1:
                    im = ax[k].imshow(layer)
                    ax[k].tick_params(labelsize=8)
                else:
                    im = ax[int(k/images_per_row),k%images_per_row].imshow(layer)
                    ax[int(k/images_per_row),k%images_per_row].tick_params(labelsize=8)
                k += 1
        
            while k < int((image.shape[0]-1)/images_per_row+1)*images_per_row:
                layer = np.full((image.shape[1],image.shape[2]),255.0)
                im = ax[int(k/images_per_row),k%images_per_row].imshow(layer)
                k += 1
            plt.tight_layout()
            plt.show()
            j += 1
        i += 1  

def image_show(images):
    images_per_row = 6
    k = 0
    images = np.array(images, dtype=np.float32)
    fig, ax = plt.subplots(nrows=int((images.shape[0]-1)/images_per_row + 1),ncols=images_per_row,
                                   figsize=(2*images_per_row, 2*int((images.shape[0]-1)/images_per_row + 1)))
    for layer in images:
        if int((images.shape[0]-1)/images_per_row + 1) == 1:            
            im = ax[k].imshow(layer)
            ax[k].tick_params(labelsize=8)
        else:
            im = ax[int(k/images_per_row),k%images_per_row].imshow(layer)
            ax[int(k/images_per_row),k%images_per_row].tick_params(labelsize=8)
        k += 1

    while k < int((images.shape[0]-1)/images_per_row+1)*images_per_row:
        layer = np.full((images.shape[1],images.shape[2]),255.0)
        im = ax[int(k/images_per_row),k%images_per_row].imshow(layer)
        k += 1
    plt.tight_layout()
    plt.show()

class save_data(threading.Thread):
    def __init__(self, filename, datalist):
        threading.Thread.__init__(self)
        self.filename = filename
        self.datalist = datalist.copy()
        
    def run(self):
        data = np.array(self.datalist)
        np.save(self.filename,data)    
                       
class vis_nn(object):
    def __init__(self, path='./vis_log/'):
        self.log_path = path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path) 
        self.nn_layers = []
        
    def append_layers(self, layers):
        self.nn_layers.append(layers)       
       
    def log_data(self, train_step, layer_name, data): 
        data = np.array(data).transpose(0,3,1,2)
        image_index = 0            
        for image in data:
            layer_index = 0

            log_path_step = self.log_path+'train_step_'+str(train_step) + '/image_index_'+str(image_index)+ '/'
            if not os.path.exists(log_path_step):
                os.makedirs(log_path_step) 

            for nn_layer in image:
                while threading.activeCount() > 20:
                    time.sleep(0.05)

                save_thread = save_data(filename = log_path_step+layer_name+'_'+str(train_step)+'_'+str(image_index)+'_'+str(layer_index),
                                        datalist = nn_layer)
                save_thread.start()
                layer_index += 1
            image_index += 1
    
    def read_data(self, train_step, layer_name, image_index, layer_index):
        log_path_step = self.log_path+'train_step_'+str(train_step) + '/image_index_'+str(image_index)+ '/'
        data = np.load(log_path_step+layer_name+'_'+str(train_step)+'_'+str(image_index)+'_'+str(layer_index))
        return data
    
    def read_nn_flow(self, train_step, image_index):
        log_path_step = self.log_path+'train_step_'+str(train_step) + '/image_index_'+str(image_index)+ '/'
        for layer in self.nn_layers:  
            layer_name = layer[0]
            layers = []
            for item in os.walk(log_path_step):
                for file in item[2]:                    
                    if file[:len(layer_name)] == layer_name:
                        image = self.read_data(train_step,layer_name,image_index,file[file.rfind('_')+1:])
                        layers.append(image)             
            print('---------------------' + layer_name + '---------------------')
            image_show(layers)         
                        
                        
            #data = read_data(train_step, layer_name,image_index)
    #def visualize_layers(self, train_step, batch_index, image_index):
        