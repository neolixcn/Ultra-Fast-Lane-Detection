import torch, os, cv2
import pdb
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import time
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from utils.common import decode_seg_color_map

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        lane_num = 4
    elif cfg.dataset == 'Bdd100k':
        cls_num_per_lane = 56#18
        lane_num = 4#14
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        lane_num = 4
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, lane_num),
                   use_aux=cfg.use_aux).cuda() # we dont need auxiliary segmentation in testing
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
    if cfg.dataset == 'CULane':

        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Bdd100k':
        row_anchor = tusimple_row_anchor
    elif cfg.dataset == 'Tusimple':

        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    data_folder = "/data/pantengteng/lane_detection/hengtong_lane_detection"# "/data/pantengteng/neolix_lane/images"#"/nfs/nas/dataset/test_set_operational_scenarios/traffic_light/images"#"/data/pantengteng/lane_detection/shanghai_lane_detection"  #"/data/pantengteng/neolix_lane"
    save_folder = os.path.join(os.getcwd(), "ufld_fish_results/hengtong")
    # save_folder = os.path.join(os.getcwd(), f"ufld_fish_results/{cfg.dataset}") #_SH
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    img_list = os.listdir(data_folder)
    total_time = 0
    for img_name in img_list:
        org_img = cv2.imread(os.path.join(data_folder, img_name))
        img_h, img_w, _ = org_img.shape
        img =cv2.resize(org_img, (800, 288))
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


        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(org_img, ppp,5,(0,255,0),-1)
        cv2.imwrite(os.path.join(save_folder, cfg.save_prefix + img_name), org_img)

        '''
        if cfg.use_aux:
            # pdb.set_trace()
            seg_predict = torch.argmax(seg_out, dim=1).squeeze(0)
            seg_img = decode_seg_color_map(seg_predict)
            seg_img = seg_img.data.cpu().numpy()
            cv2.imwrite(os.path.join(save_folder, cfg.save_prefix +"seg"+ img_name), seg_img)
        '''

    print(total_time / len(img_list))