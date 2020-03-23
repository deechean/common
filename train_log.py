#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:54:08 2019

@author: Deechean
"""

import os 
import time

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
        with open(self.log_path+var_name,'a+',encoding='utf-8') as f:
            for x in datalist:
                f.write(str(x) + '\n') 
                
    def SaveToFile(self):
        for var_name in self.log_dic:
            self.saveEvalData(var_name, self.log_dic[var_name]) 
            self.log_dic[var_name].clear()
    
    def write_file(self, filename, data):
        with open(self.log_path+filename,'a+',encoding='utf-8') as f:
            for item in data:
                f.write(str(item) + '\n') 
             
    def readlog(self, var_name):
        parameterlist = []
        with open(self.log_path+var_name,'r',encoding='utf-8') as f:
            while True: 
                line = f.readline()
                if line == '':
                    break
                if line.find(' '+var_name+':') > 0:
                    pos = line.find('global step:')
                    if pos > 0: 
                        line = line[line.find('global step:')+len('global step:'):]
                        step  = int(line[:line.find(',')])
                        value = line[line.find(var_name+':')+len(var_name+':'):line.find('\n')]
                        try:
                            value = float(value)
                        except ValueError:
                            if value[0] == '[':
                                value = value[1:]
                                itemlist = []
                                while value.find(",") > 0: 
                                    item = float(value[:value.find(",")])
                                    itemlist.append(item)
                                    value = value[value.find(",")+1:]
                                item = float(value[:-1])
                                itemlist.append(item)
                            value = itemlist                                    
                    parameterlist.append([step, value]) 
        return parameterlist