export CUDA_VISIBLE_DEVICES=2,3
export NGPUS=2
export OMP_NUM_THREADS=4 # you can change this value according to your number of cpu cores


# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/culane.py
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 29501 train.py configs/neolix.py --distributed True --epoch 50 --batch_size 32 --learning_rate 0.1 --backbone '18' --use_aux True
#python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 29501 train.py configs/tusimple.py --distributed True --epoch 100 --batch_size 32 --learning_rate 0.1 --backbone '18'
# --finetune "/data/pantengteng/tensorboard_logs/20210105_210507_lr_1e-01_b_8/ep093.pth"
#--finetune "/data/pantengteng/tensorboard_logs/20210110_173730_lr_1e-01_b_2/ep032.pth"
# python train.py configs/tusimple.py
