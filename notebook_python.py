""" Python学习笔记 """
#1. numpy和pandas是用c语言写的，pandas基于numpy;
#2. 文件头部指定编码方式，UTF-8是Unicode的一种实现方式;
#3. Python中默认的编码格式是ASCII格式，在没修改编码格式时无法正确打印汉字。

# Anaconda：常用命令
!conda info #系统信息
!conda list #查看所有安装包
!pip list #查看所有安装的包

# 例子：安装包
!pip install tushare #安装包
import tushare as ts
# 升级
!pip -m pip install --upgrade pip

# 万得接口，wind
import WindPy as api
api.w.start()
api.w.isconnected()

# 设置工作目录路径
import os
os.getcwd() #获取当前工作目录
os.chdir('G:\\xx\\yy') #设置工作目录 



