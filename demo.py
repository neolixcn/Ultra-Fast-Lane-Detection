import torch
import os
import cv2
import pdb
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import time
import scipy.special
import tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from utils.common import decode_seg_color_map

if __name__ == "__main__":
    with_label = True

    # "/data/pantengteng/lane_detection_test/hengtong_sunny_noon"
    # "/data/pantengteng/lane_detection/shanghai_lane_detection"
    # "/data/pantengteng/lane_detection_test/snow_lane_detection"
    data_folder = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lane_detection/neolix_lane_fisheye/lane/"
    lane_list_file = "val.txt"
    save_folder = os.path.join(
        os.getcwd(), "ufld_fish_results/id013")

    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101',
                            '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        lane_num = 4
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Bdd100k':
        cls_num_per_lane = 56  # 18
        lane_num = 4  # 14
        row_anchor = tusimple_row_anchor
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        lane_num = 4
        row_anchor = tusimple_row_anchor
    elif cfg.dataset == 'neolix':
        cls_num_per_lane = 18
        lane_num = 4
        row_anchor = culane_row_anchor
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num+1, cls_num_per_lane, lane_num),
                     use_aux=cfg.use_aux).cuda()  # we dont need auxiliary segmentation in testing
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        # transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    if lane_list_file:
        with open(os.path.join(data_folder, lane_list_file), "r") as f:
            txt_list = f.readlines()
        img_list = [line.split()[0] for line in txt_list]
        if with_label:
            label_list = [line.split()[1] for line in txt_list]
    else:
        img_list = os.listdir(data_folder)
    total_time = 0
    for img_index, img_name in enumerate(img_list):
        org_img = cv2.imread(os.path.join(data_folder, img_name))
        if len(img_name.split("/")) > 1:
            save_img_name = img_name.split(
                "/")[-2] + "_" + img_name.split("/")[-1]
        else:
            save_img_name = img_name

        base_img = org_img.copy()
        img_h, img_w, _ = org_img.shape
        img = cv2.resize(org_img, (800, 288))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        in_img = img_transforms(img).unsqueeze(0).cuda()

        start_time = time.time()
        with torch.no_grad():
            if cfg.use_aux:
                cls_out, seg_out = net(in_img)
            else:
                cls_out = net(in_img)
            end_time = time.time()
        total_time += end_time - start_time

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        out_j = cls_out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]

        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        bk_img = np.zeros_like(org_img)
        if with_label:
            label = cv2.imread(os.path.join(
                data_folder, label_list[img_index]), 0)
            org_img[label > 0] = [0, 255, 0]
            # cv2.imwrite(os.path.join(save_folder, cfg.save_prefix +"_temp_"+ save_img_name), bk_img)
        name_list = ["left2", "left1", "right1", "right2"]

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                x = []
                y = []
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        # 拟合
                        x.append(
                            int(out_j[k, i] * col_sample_w * img_w / 800) - 1)
                        y.append(
                            int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(
                            img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                        # cv2.circle(bk_img, ppp, 5, (0, 0, 0), -1)

                        # 车道线点
                        cv2.circle(base_img, ppp, 5, (0, 255, 0), -1)
                x = np.array(x)
                y = np.array(y)
                res = np.polyfit(x, y, 3)
                func = np.poly1d(res)
                new_x = np.linspace(x[0], x[-1], 100)
                new_y = func(new_x)
                for j in range(new_x.shape[0] - 1):
                    cv2.line(org_img, (int(new_x[j]), int(new_y[j])), (int(
                        new_x[j+1]), int(new_y[j+1])), (0, 0, 255), 8, 4)
                cv2.putText(org_img, name_list[i], (int(
                    new_x[j+1]), int(new_y[j+1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # org_img = cv2.addWeighted(bk_img, 0.2, org_img, 0.8, 0, dtype=cv2.CV_32F)
        fit_save_folder = os.path.join(save_folder, 'fit')
        if not os.path.exists(fit_save_folder):
            os.makedirs(fit_save_folder)
        cv2.imwrite(os.path.join(fit_save_folder, save_img_name), org_img)
        point_save_folder = os.path.join(save_folder, 'point')
        if not os.path.exists(point_save_folder):
            os.makedirs(point_save_folder)
        cv2.imwrite(os.path.join(point_save_folder, save_img_name), base_img)
        # pdb.set_trace()

    print(total_time / len(img_list))
