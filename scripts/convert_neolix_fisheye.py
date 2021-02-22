import os
import cv2
import tqdm
import math
import numpy as np
import pdb
import json, argparse

SHOWFLAG = True

def calc_k(line):
    '''
    Calculate the direction of lanes
    '''
    line_x = [point["x"] for point in line]
    # ployfit无法拟合x=a的直线，单独考虑
    # if line_x[0] == line_x[-1]:
    #     return math.pi / 2

    line_y = [point["y"] for point in line]
    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10                                          # if the lane is too short, it will be skipped

    p = np.polyfit(line_x, line_y,deg = 1)
    rad = np.arctan(p[0])
    
    return rad
def draw(im,line,idx,show = SHOWFLAG):
    '''
    Generate the segmentation label according to json annotation
    '''
    line_x = [point["x"] for point in line]
    line_y = [point["y"] for point in line]
    pt0 = (int(line_x[0]),int(line_y[0]))
    if show:
        cv2.putText(im,str(idx),(int(line_x[len(line_x) // 2]),int(line_y[len(line_x) // 2]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = idx * 60
        
    for i in range(len(line_x)-1):
        cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),(idx,),thickness = 16)
        pt0 = (int(line_x[i+1]),int(line_y[i+1]))

def get_neolix_list(root, label_file):
    '''
    Get all the files' names and line points from the json annotation
    '''
    label_path = os.path.join(root, label_file)
    with open(label_path, "r") as f:
        label_content = f.readlines()
    names = []
    labels = []
    for line in label_content:
        if "url" in line:
            continue
        content = line.split()
        url, img_name, str_result = content
        result = json.loads(str_result)
        sub_folder = url.split("/")[-2]
        # 创建labels下的subfolder
        if not os.path.exists(os.path.join(root, "labels", sub_folder)):
            os.makedirs(os.path.join(root, "labels", sub_folder))
        names.append("images/" + sub_folder + "/" + img_name)
        label = []
        pdb.set_trace()
        for lane in result["result"][0]["elements"]: 
            # skip sub_lines and road edges
            if "sub_ID" in lane["attribute"] and lane["attribute"][ "sub_ID"] is not None:
                continue
            if lane["attribute"][ "single_line" ] == "road_edge":
                continue
            # label.append(sorted(lane["points"], key=lambda x:x["y"]))
            label.append(lane["points"])
        labels.append(label)

    return names,labels

def generate_segmentation_and_train_list(root, line_txt, names):
    """
    The lane annotations of the Neolix dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    tbar = tqdm.tqdm(range(len(line_txt)))
    for i in tbar:
        tbar.set_postfix(img=names[i][7:])
        lines =  line_txt[i]
        ks = np.array([calc_k(line) for line in lines])             # get the direction of each lane

        k_neg = ks[ks<0].copy()
        k_pos = ks[ks>0].copy()
        k_neg = k_neg[k_neg != -10]                                      # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()
        k_pos.sort()

        # name = "images/"
        label_path = "labels/" + names[i][7:-3]+'png'
        #label = np.zeros((1080,1990),dtype=np.uint8)
        label = np.zeros((720,1280),dtype=np.uint8)
        bin_label = [0,0,0,0]
        if len(k_neg) == 1:                                           # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[1] = 1
        elif len(k_neg) == 2:                                         # for two lanes in the left
            which_lane = np.where(ks == k_neg[1])[0][0]
            draw(label,lines[which_lane],1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[0] = 1
            bin_label[1] = 1
        elif len(k_neg) > 2:                                           # for more than two lanes in the left, 
            which_lane = np.where(ks == k_neg[1])[0][0]                # we only choose the two lanes that are closest to the center
            draw(label,lines[which_lane],1)
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            bin_label[0] = 1
            bin_label[1] = 1

        if len(k_pos) == 1:                                            # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],3)
            bin_label[2] = 1
        elif len(k_pos) == 2:
            which_lane = np.where(ks == k_pos[1])[0][0]
            draw(label,lines[which_lane],3)
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],4)
            bin_label[2] = 1
            bin_label[3] = 1
        elif len(k_pos) > 2:
            which_lane = np.where(ks == k_pos[-1])[0][0]
            draw(label,lines[which_lane],3)
            which_lane = np.where(ks == k_pos[-2])[0][0]
            draw(label,lines[which_lane],4)
            bin_label[2] = 1
            bin_label[3] = 1
        
        cv2.imwrite(os.path.join(root,label_path),label)

        #with open(os.path.join(root,'train_gt.txt'),'a') as gt_fp:
        #    gt_fp.write(names[i] + ' ' + label_path + ' '+' '.join(list(map(str,bin_label))) + '\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Neolix dataset')
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()

    # filter = ["solid white", "solid yellow", "broken white", "broken yellow", "double solid white", "double solid yellow", "solid & broken white", "solid & broken yellow"]

    # training set
    names,line_txt = get_neolix_list(args.root, "车道线标注总_共2505帧.txt")#"label.txt")
    # generate segmentation and training list for training
    generate_segmentation_and_train_list(args.root, line_txt, names)


