# Ultra-Fast-Lane-Detection
PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)", which is accepted by ECCV2020.


![alt text](vis.jpg "vis")

The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN) and [Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark).


# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started

## Configs
Config the file `configs/XXX.py` according to your environment.Necessary parameters to be configed:  
 - `data_root` is the path of your training dataset for scheme1.
 - `val_data_root` is the path of validating dataset for scheme2.
 - `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***
 - `finetune` is the model to be finetuned with.
 - `resume` is the checkpoint saved in previous training to be resumed in current training.

 Besides config style settings, we also support command line style one. By change the you can override a setting like
```Shell
python train.py configs/path_to_your_config --batch_size 8
```
The **batch_size** will be set to 8 during training.

## Training
***
For single gpu training, run
```Shell
sh training_bdd.sh
```
or
```Shell
python train.py configs/path_to_your_config
```

For multi-gpu training, run
```Shell
sh launch_training.sh
```
or
```Shell
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
```

***

To visualize the log with tensorboard, run

```Shell
tensorboard --logdir log_path --bind_all
```

# Evaluation




# Test and Visualization

We provide a script to visualize the detection results. Run the following commands to visualize on testing set.
```Shell
python demo.py configs/bdd100k.py --test_model path_to_pth

```
You can also use shell script as follows.
```
sh test_demo.sh
```

# Assistant Scripts
Some assistant scripts to prepare neolix dataset is provided:
 - convert_neolix_fisheye.py and convert_neolix.py are to convert neolix fisheye data and 3mm data that labeled by baidu to proper format.
 - split_train_val.py is to divide dataset into training set and validating set by generate seperated list files.
