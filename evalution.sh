import os

export CUDA_VISIBLE_DEVICES=0,1 
export NGPUS=4
export OMP_NUM_THREADS=2

# tusimple预训练模型在tusimple测试集上指标
#python test.py configs/tusimple.py --test_model '/data/pantengteng/tensorboard_logs/lanxin/20210121_123252_lr_1e-01_b_32/ep019.pth' --test_work_dir '/home/pantengteng/Programs/Programs_Lanxin/tmp/tusimple1'
#python test.py configs/bdd100k.py --test_model '/data/pantengteng/tensorboard_logs/20210126_175045_lr_1e-01_b_32/ep049.pth' --test_work_dir '/home/pantengteng/Programs/Programs_Lanxin/tmp/evalution/005'
#python test.py configs/bdd.py --test_model '/data/pantengteng/tensorboard_logs/20210204_165841_lr_1e-01_b_32/ep042.pth' --test_work_dir '/home/pantengteng/Programs/Programs_Lanxin/tmp/evalution/013'
#python test.py configs/tusimple.py --test_model '/data/pantengteng/tensorboard_logs/lanxin/20210223_104002_lr_1e-01_b_32/ep050.pth' --test_work_dir '/home/pantengteng/Programs/Programs_Lanxin/tmp/evalution/tusimple'


python test.py configs/neolix.py --test_model '/nfs/neolix_data1/lanxin_temp/tensorboard_logs/20210514_101903_lr_1e-02_b_32/ep045.pth' --test_work_dir '/home/liuwanqiang/leilanxin/tmp'
