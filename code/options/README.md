# Configurations
- Use **json** files to configure options.
- Convert the json file to python dict.
- Support `//` comments and use `null` for `None`.

### train_spsr.json
```c++
{
  "name": "SPSR" 
  , "use_tb_logger": true // use tensorboard_logger, 
  , "model":"spsr" // model type
  , "scale": 4 // scale factor for SR
  , "gpu_ids": [0,1] // specify GPUs, actually it sets the `CUDA_VISIBLE_DEVICES`

  , "datasets": { // configure the training and validation datasets
    "train": { // training dataset configurations
      "name": "DIV2K" // dataset name
      , "mode": "LRHR" // dataset mode, ref: `https://github.com/Maclory/SPSR/codes/data/__init__.py`
      , "dataroot_HR": "/mnt/4/jzy/dataset/SR_dataset/DIV2K800/DIV2K800_HR.lmdb" // HR data root please modify to your own HR root before training.
      , "dataroot_LR": "/mnt/4/jzy/dataset/SR_dataset/DIV2K800/DIV2K800.lmdb" // LR data root 
      , "subset_file": null // use a subset of an image folder
      , "use_shuffle": true // shuffle the dataset
      , "n_workers": 16 // number of data load workers
      , "batch_size": 30
      , "HR_size": 128 
      , "use_flip": true // whether use horizontal and vertical flips
      , "use_rot": true // whether use rotations: 90, 190, 270 degrees
    }
    , "val": { // validation dataset configurations
      "name": "Set14"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/4/jzy/dataset/SR_dataset/Set14/Set14_sub"
      , "dataroot_LR": "/mnt/4/jzy/dataset/SR_dataset/Set14/Set14_sub_bicLRx4"
    }
  }

  , "path": {
    "root": "/mnt/4/SPSR/cya_sr/cya_BasicSR/release" //modify to you own root path
    // , "resume_state": "../experiments/SPSR_0311/training_state/25000.state"
    , "pretrain_model_G": "../experiments/pretrained_models/RRDB_PSNR_x4.pth"
  }

  , "network_G": { // configurations for the network G
    "which_model_G": "spsr_net" 
    , "norm_type": null //  
    , "mode": "CNA" // Convolution mode: CNA for Conv-Norm_Activation
    , "nf": 64 // number of features for each layer
    , "nb": 23 // number of blocks
    , "in_nc": 3 // input channels
    , "out_nc": 3 // output channels
    , "gc": 32 // grouwing channels, for Dense Block
    , "group": 1 // convolution group, for ResNeXt Block
  }

  , "train": {
    "lr_G": 1e-4
    , "lr_G_grad": 1e-4
    , "weight_decay_G": 0
    , "weight_decay_G_grad": 0
    , "beta1_G": 0.9
    , "beta1_G_grad": 0.9
    , "lr_D": 1e-4
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [50000, 100000, 200000, 300000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 1e-2 // weight for pixel loss
    , "feature_criterion": "l1"
    , "feature_weight": 1 // weight for feature loss
    , "gan_type": "vanilla"
    , "gan_weight": 5e-3 // weight for gan loss
    , "gradient_pixel_weight": 1e-2 // weight for gradient gan loss
    , "gradient_gan_weight": 5e-3 // weight for gradient gan loss
    , "pixel_branch_criterion": "l1" // criterion for the gradient branch
    , "pixel_branch_weight": 5e-1
    , "Branch_pretrain" : 1 // whether use some iterations to pretrain the gradient branch
    , "Branch_init_iters" : 5000 // iterations that used to pretrain the gradient branch

    , "manual_seed": 9
    , "niter": 5e6
    , "val_freq": 5e3
  }

  , "logger": { // logger configurations
    "print_freq": 200
    , "save_checkpoint_freq": 5e3
  }
}
```
