# DATA
dataset='Tusimple'
data_root = '/data/pantengteng/TuSimple'#None
val_dataset= None
val_data_root = None

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# VAL
val = False
val_batch_size = 2#4

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
cls_loss_w  = 1
seg_loss_w = 1# -1 #2
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/data/pantengteng/tensorboard_logs/lanxin'#None

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4