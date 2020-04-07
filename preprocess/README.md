# Preprocess

## Dependencies and Installation

Please install required modules as follows:

```bash
$ pip install -r requirements.txt
```

## Generate Datasets

1. Copy all the datasets you want to process into the `../data/GT` folder.

2. Edit `generate_dataset.py`, change the `dataDir` path, `saveDir` path and `scale` as follows:

```python
dataDir = '../data/GT' # path to GT folder
saveDir = '../data/dataset' # path to save results
scale = '4' # up sample scale and picture size scale
```

3. Start generating datasets:

```bash
$ python generate_dataset.py
```

4. The datasets of LR, HR and Bicubic-upsampled versions will be saved into a sub-folder which is named from the ground-truth datasets and is stored into `saveDir`.

## Ways to Speed up I/O

- Change your hard disk from HDD to SSD
- Change images in the dataset to smaller sub-image slices
- Change the format of the dataset to `.lmdb`

## Generate Sub-Images

1. Edit the `extract_subimgs_single.py` and change the `input_folder` and `save_folder` path as follows:

```python
input_folder = os.path.join('../data','dataset','XXX','HR')
save_folder = os.path.join('../data','dataset','XXX_sub','HR')
```

Tips: `XXX` is the name of your dataset, `XXX_sub` is the name of your sub-image folder to be stored.

2. Start generating sub-images:

```bash
$ python extract_subimgs_single.py
```

3. For **LR images**, the procedure is the same as above.

Warning: If you want to create sub-images for both HR and LR images, please edit the values of `crop_sz`, `step` and `thres_sz` in order to match LR sub-images and the HR counterparts.

```python
crop_sz = 480
step = 240
thres_sz = 48
```

## Generate LMDB

1. Edit the `create_lmdb.py` and change the `img_folder` path and `lmdb_save_path` as below:

```python
img_folder = os.path.join('../data','dataset','XXX_sub','HR','*')  # glob matching pattern
lmdb_save_path = os.path.join('../data','dataset','XXX_sub_HR.lmdb')  # must end with lmdb
```

Tips: `XXX_sub` is the name of the folder where your sub-images are stored, `XXX_sub_HR.lmdb` is the name of the generated lmdb file which must end with '.lmdb'.

2. Start generating lmdb files:

```bash
$ python create_lmdb.py
```

3. For **LR images**, the procedure is the same as above.

## Reference

`create_lmdb.py`, `extract_subimgs_single.py`, `generate_dataset.py` are modified from [BasicSR](https://github.com/xinntao/BasicSR).
