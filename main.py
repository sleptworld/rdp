import sys
import platform
import os
import argparse
from rd import *
"""
TEST RUOMU
"""

if __name__ == "__main__":
# 检测运行库
    system = platform.system()
    Libraries = ['numpy','cv2','imutils','scipy']
    ALL = os.popen("pip3 list").read()
    for name in Libraries:
        if name in ALL:
            print("OK")
        else:
            print("MODULES %s HAVE NOT INSTALLED" %name)
    
    args = argparse.ArgumentParser(description='对一张滴谱图进行处理')
    args.add_argument('-p','--path',type=str,required=True,help='传入图片地址')
    args.add_argument('-o','--outpath',type=str,default='./',help='处理后图片的存储位置，默认当前目录下')
    args.add_argument('-t','--threshold',type=tuple,default=(50,100),help='Canny查找边缘的阈值，默认为（50，100）')
    args.add_argument('-w','--width',type=float,required=True,help='左侧第一个点的实际尺寸 如5；无需输入单位')
    args.add_argument('--size',type=float,default=120,help='排除较小噪点的尺寸 默认值为120')
    args.add_argument('-s','--save',type=bool,default=True,help='是否保存中途处理过程，默认保存')
    a = args.parse_args()

    path = a.path
    outpath = a.outpath
    threshold = a.threshold
    save = a.save
    size = a.size
    width = a.width

    rdp = ORdp(path,width=width)

    rdp.init(r1=threshold[0],r2=threshold[1])

    rdp.foundCounters(path=path,size=size)

    if save:
        rdp.saveGrayAndbw(path=path)

    rdp.inch()

    print("=======================")