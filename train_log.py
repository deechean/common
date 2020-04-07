#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:54:08 2019

@author: Deechean
"""

import os 
import time
import threading

def isIter(variable):
    try:
        iter(variable)
        return True
    except:
        return False
    
def strlize(variable):   
    if isIter(variable):
        valuestr = '['
        for item in variable:
            valuestr += strlize(item) + ','
        valuestr = valuestr[:len(valuestr)-1]+']'
    else:
        valuestr = str(variable)
    return valuestr

def readiter(strValue):
    if len(strValue) > 0:
        i=0
        while strValue[i] == '[':
            i+=1
        pos = strValue.find(']'*i)
        value = []
        if strValue[0] == '[':                      
            value_list = readiter(strValue[1:pos+i-1])          
            value.append(value_list)
            value_list = readiter(strValue[pos+i+1:])
            for item in value_list:
                value.append(item)            
        else:
            value_tmp = strValue.split(',')
            for item in value_tmp:
                value.append(float(item))
                #print(float(item))
        return value
    else:
        return []

class save_log(threading.Thread):
    def __init__(self, filename, datalist):
        threading.Thread.__init__(self)
        self.filename = filename
        self.datalist = datalist.copy()
        
    def run(self):
        with open(self.filename,'a+',encoding='utf-8') as f:
            for x in self.datalist:
                f.write(str(x) + '\n')    
        #print('File saved.')
    
class train_log(object):
    def __init__(self,path='log/'):
        self.log_path = path
        self.log_variable = []
        self.log_dic = {}    
    
    def add_log(self, var_name, globalstep, value):  
        if not(var_name in self.log_variable):
            self.log_dic[var_name] = []
            self.log_variable.append(var_name)
        valuestr = strlize(value)        
        self.log_dic[var_name].append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+ ', global step: '+ str(globalstep) + ', '+ var_name +':'+valuestr)
        
    def saveEvalData(self,var_name, datalist):
        filename = self.log_path+var_name
        save_thread = save_log(filename, datalist)
        save_thread.start()
        #print('Start a thread to save.')
                    
    def SaveToFile(self):
        for var_name in self.log_dic:
            self.saveEvalData(var_name, self.log_dic[var_name]) 
            self.log_dic[var_name].clear()
            
    def write_file(self, filename, data):
        with open(self.log_path+filename,'a+',encoding='utf-8') as f:
            for item in data:
                f.write(str(item) + '\n') 
                
    def readlog(self, var_name, maxrecord=10000):
        parameterlist = []
        with open(self.log_path+var_name,'r',encoding='utf-8') as f:
            print('file open success')
            i = 0
            while (True and i <maxrecord): 
                line = f.readline()
                i += 1
                if line == '':
                    break
                if line.find(' '+var_name+':') > 0:
                    pos = line.find('global step:')
                    if pos > 0: 
                        line = line[line.find('global step:')+len('global step:'):]
                        step  = int(line[:line.find(',')])
                        value = line[line.find(var_name+':')+len(var_name+':'):line.find('\n')]
                        value_list = readiter(value)[0]                       
                    parameterlist.append([step, value_list]) 
        return parameterlist