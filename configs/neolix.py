# DATA
dataset= 'neolix'
data_root = "/data/pantengteng/neolix_lane_fisheye/lane/"

val_dataset='neolix'
val_data_root='/data/pantengteng/neolix_lane_fisheye/lane/'

# TRAIN
epoch = 50#100
batch_size = 4#8#4#32
optimizer = 'SGD' #'Adam'#'Adam'#['SGD','Adam']
learning_rate = 0.1 #0.01 #0.001
weight_decay = 1e-4
momentum = 0.9

scheduler =  'cos' #['multi', 'cos']
steps = [25] #[25,38]
gamma  = 0.1
warmup = 'linear' #None#
warmup_iters = 200#100#695

# VAL
val = True
val_batch_size = 4

# NETWORK
use_aux = True
seg_class_num = 19
griding_num = 200
backbone = '101'#'18'#

# LOSS
cls_loss_w  = 1
seg_loss_w = 1# -1 #2
sim_loss_w = 0 #0.2#0.0# 0.1 relation_loss
shp_loss_w = 0 #0.1#0.0# 0.1 relation_dis

# EXP
note = ''

log_path = "/data/pantengteng/tensorboard_logs"# "/home/pantengteng/Programs/tensorboard_logs"

# FINETUNE or RESUME MODEL PATH
finetune = None #"/data/pantengteng/tensorboard_logs/20210105_210507_lr_1e-01_b_8/ep093.pth" #None #"./tusimple_18.pth"#None #None #./culane_18.pth"
resume = None #"/data/pantengteng/tensorboard_logs/20210105_210507_lr_1e-01_b_8/ep093.pth"#None
 
# TEST
test_model = "/data/pantengteng/tensorboard_logs/20210105_210507_lr_1e-01_b_8/ep099.pth"
test_work_dir = None
save_prefix = "new_2021"

num_lanes = 4 #14