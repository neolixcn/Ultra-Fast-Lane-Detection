import torch
import os
import cv2 as cv
import json
import pdb
import numpy as np

#将neolix测试集标签转换为与tusimple相同格式的json文件，便于使用tusimple自带评估方法
from data.constant import tusimple_row_anchor

path="/nfs/neolix_data1/neolix_dataset/test_dataset/lane_detection/neolix_lane_fisheye/lane/labels"
json_path="/nfs/neolix_data1/neolix_dataset/test_dataset/lane_detection/neolix_lane_fisheye/lane/test_label.json"

def find_mid_point(list,num):
    count=0
    sum=0
    for i in range(len(list)):
        if num==list[i]:
            count+=1
            sum+=i
    return int(sum/count)

if __name__=="__main__":
    for i in os.listdir(path):
        for j in os.listdir(os.path.join(path,i)):
            lanes=[]
            img=cv.imread(os.path.join(path,i,j),0)
            img=cv.resize(img,(800,288),interpolation=cv.INTER_NEAREST)
            for num in range(4):
                lane=[]
                for row_anchor_num in range(len(tusimple_row_anchor)):
                    if num+1 not in img[tusimple_row_anchor[row_anchor_num]]:
                        lane.append(-2)
                    else:
                        lane.append(find_mid_point(img[tusimple_row_anchor[row_anchor_num]],num+1))
                lanes.append(lane)
            a={"lanes":lanes,
            "h_samples":[160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710],
            "raw_file":"images/"+i+"/"+j
            }
            with open(json_path,"a") as f:
                json.dump(a,f)
                f.write("\n")
            #pdb.set_trace()