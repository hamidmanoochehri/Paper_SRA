# Project Title

A brief description of your project and its purpose.

## Table of Contents
- [Stain Reconstruction Augmentation](#stain_reconst_augm)
- [Needed Packages](#reqs)
- [Usage](#usage)
- [Structure](#structure)
- [Contributing](#contributing)
- [License](#license)

## Dependencies
torch       = 2.0.1+cu118 \
torchvision = 0.15.2+cu118 \
pillow      = 9.4.0 \
numpy       = 1.26.2 \
pandas      = 1.2.3 \
sklearn     = 1.2.2 \
tensorboard = 2.15.1 \

## Usage
Example usage (Pretraining):
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_moco.py \
    --mode 'train' \
    --rgb-he-wrgb-dist-Hmax 'uniform' \
    --rgb-he-wrgb-params-Hmax 0.1 2.5 \
    --rgb-he-wrgb-dist-Emax 'uniform' \
    --rgb-he-wrgb-params-Emax 0.1 2.5 \
    --batch-size 512 \
    --epochs 400 \
    --arch 'resnet50' \
    --moco-m-cos --crop-min=.2 \
    --dist-url 'tcp://xxxxxx:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --data /data_folder_in_imagefolder_format
echo "Pretraining completed"
```

Example usage (Classification):
```bash
CUDA_VISIBLE_DEVICES=$gpu python main_lincls.py \
    --data /data_folder_in_imagefolder_format \
    --pretrained /path_to_pretrained_model \
    --epochs 30 \
    --arch resnet50 \
    --iter_per_epoch 400 \
    --balanced_sampler \
    --batch-size 64 \
    --lr 0.1 \
    -j 32
echo -e "\n\Classification completed"
```

# Structure
* Only required files for implementation of SRA are listed \
```
sra_v4.py                    : Stain Reconstruction Augmentation (SRA) main code \
tcgakirc_adaptive_params.txt : Calculated adaptive parameters for TCGA KIRC slides \
code_snapshotting.py         : Takes copy of the required codes to the results folder at the beginning og each experiment along with unique experiment number \
main_moco.py                 : Pretraining main code (run this) \
main_lincls.py               : Classification main code (run this) \
moco/ \
    builder_v1.py            : \
    loader.py                : \
    optimizer.py             : \
pytorch_balanced_sampler/     : Balanced sampler code for classification (obtained from TODO) \
    sampler.py \
    utils.py \
README.md                    : README file (this file) \
```
