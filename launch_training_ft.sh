dexport CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export OMP_NUM_THREADS=4 # you can change this value according to your number of cpu cores


# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/culane.py
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 29678 train.py configs/neolix.py --distributed True --learning_rate 0.1 --epoch 50 --backbone "18" --batch_size 32 --finetune "/nfs/neolix_data1/lanxin_temp/tensorboard_logs/lane/20210202_182154_lr_1e-01_b_32/ep049.pth" #--resume "/nfs/neolix_data1/lanxin_temp/tensorboard_logs/20210517_135839_lr_1e-01_b_32/ep031.pth" 
#--resume "/data/pantengteng/tensorboard_logs/20210204_165841_lr_1e-01_b_32/ep042.pth"
# --width 1280 --height 720 
#--finetune "/data/pantengteng/tensorboard_logs/20210105_210507_lr_1e-01_b_8/ep093.pth"
