
# CUDA_VISIBLE_DEVICES=1 python demo_test.py configs/bdd100k.py --backbone '101' --test_model "/data/pantengteng/tensorboard_logs/20210105_210507_lr_1e-01_b_8/ep093.pth" --save_prefix "zhangjiang"#"bdd_210106"

CUDA_VISIBLE_DEVICES=0 python demo.py configs/bdd100k.py --backbone '101' --test_model "/data/pantengteng/tensorboard_logs/20210115_194443_lr_1e-01_b_2/ep057.pth" --save_prefix "hengtong_200_57_"#"bdd_210106"
