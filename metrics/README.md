# Evaluation Toolbox

## Dependencies

- Python3
- Python packages in `requirements.txt`
- MATLAB Engine API for Python

## Installation

### A. Install MATLAB Engine API for Python

#### a. Windows

##### 1. Find your MATLAB root

```bash
where matlab
```

For example:

```bash
C:\Program Files\MATLAB\R2019b\bin\matlab.exe
```

##### 2. Go to the Python API folder

```bash
cd matlabroot\extern\engines\python
```

For example:

```bash
cd C:\Program Files\MATLAB\R2019b\extern\engines\python
```

##### 3. Install MATLAB Engine API for Python

```bash
python setup.py install
```

#### b. Ubuntu

##### 1. Find your MATLAB root

```bash
sudo find / -name MATLAB
```

For example:

```bash
# default MATLAB root
/usr/local/MATLAB/R2016b
```

##### 2. Go to Python API folder

```bash
cd matlabroot/extern/engines/python
```

For example:

```bash
cd /usr/local/MATLAB/R2016b/extern/engines/python
```

##### 3. Install MATLAB Engine API for Python

```bash
python setup.py install
```

### B. Install Required Modules

```bash
pip install -r requirements.txt
```

### C. Download the PI Evaluating Model

You can download the PI evaluating model from [Google Drive](https://drive.google.com/open?id=1c4EbfI6X4KzCiyg1H7TA6rJY-HVgvS3q) or [Baidu Drive](https://pan.baidu.com/s/1bcDiD07aTUO2THmEXZJaiA) (extraction code muw3) and place `model.mat` into `MetricEvaluation/utils/sr-metric`.

## Usage

The scripts will calculate the values of the following evaluation metrics: [`'MA'`](https://github.com/chaoma99/sr-metric), [`'NIQE'`](https://github.com/csjunxu/Bovik_NIQE_SPL2013), [`'PI'`](https://github.com/roimehrez/PIRM2018), `'PSNR'`, [`'SSIM'`](https://ece.uwaterloo.ca/~z70wang/research/ssim), `'MSE'`, `'RMSE'`, `'MAE'`, [`'LPIPS'`](https://github.com/richzhang/PerceptualSimilarity). Note that the `'SSIM'` values are calculated by `ssim.m`, the matlab code including the suggested downsampling process available in this [link](https://ece.uwaterloo.ca/~z70wang/research/ssim). 

### Configurations

#### a. Manual

Manual modification is recommended when you want to evaluate only one SR method. You can edit `Configuration.yml` as follows:

```yaml
Pairs:
  Dataset:
    - Set5
    - Set14
  GTFolder: 
    - ../data/GT/Set5/HR
    - ../data/GT/Set14/HR
  SRFolder:
    - ../data/SPSR_Paper/Set5
    - ../data/SPSR_Paper/Set14
Name: Test
Echo: True
```

- `Pairs`: The orders of the `Dataset`, `GTFolder` and `SRFolder` must match each other.
- `Dataset`: The Datasets that need to be evaluated.
- `GTFolder`: The folder path of ground-truth images.
- `SRFolder`: The folder path of SR images.
- `Name`: The evaluation's name.
- `Echo`: Whether to echo scores while evaluating or not.

#### b. Bash

We also provide `generate_configuration.py` to generate configuration files automatically when you have multiple SR methods to evaluate.

1. Put SR folders of different methods into `your_SR_Folder` and put the ground-truth folders into `your_GT_Folder`. 

2. Edit the `MethodDict` values. The key is the method and the value is the list of datasets you want to evaluate.

```python
MethodDict=dict()
MethodDict['EnhanceNet']=['BSD100', 'General100', 'Set14']
MethodDict['SRGAN']=['BSD100', 'General100', 'Set14', 'Set5', 'Urban100']
```

3. Edit the `dataDict` values. 

```python
fileName = method+'.yml'
dataDict=dict()
dataDict['Pairs']=dict()
dataDict['Pairs']['Dataset']=MethodDict[method]
dataDict['Pairs']['SRFolder']=[]
dataDict['Pairs']['GTFolder']=[]
dataDict['Name']=evaluation_name
dataDict['Echo']=True
for dataset in MethodDict[method]:
    dataDict['Pairs']['SRFolder'].append(str(os.path.join('your_SR_Folder',method,dataset)))
    dataDict['Pairs']['GTFolder'].append(str(os.path.join('your_GT_Folder','GT',dataset,'HR')))
```

- `dataDict['Name']` is the evaluation's name.
- `dataDict['Echo']` is a bool value to control whether to output scores to the terminal while evaluating.
- `your_SR_Folder` is the folder where your SR results are stored.
- `your_GT_Folder` is the folder where your GT datasets are stored.

4. Start generating the configuration file:

```bash
$ python generate_configuration.py
```

### Evaluation

#### a. Manual

Run `evaluate_sr_results.py`

```bash
usage: python evaluate_sr_results.py [-h] YAML

positional arguments:
  YAML        configuration file

optional arguments:
  -h, --help  show this help message and exit
```

For example:

```bash
python evaluate_sr_results.py Configuration.yml
```

#### b. Bash

If you use the bash configuration method, all of the generated configuration files will be saved in the `Configuration` folder. At the same time, a bash file named  `Run.bash` will be generated to run the whole configurations. You can start evaluation by:

```bash
bash Run.bash
```

### Results

The results will be generated in the `../evaluate/` folder as follows:

- `Name-%Y%m%d%H%M%S.log`: Log files of evaluation.
- `Name-%Y%m%d%H%M%S`: Folders to store evaluation data. `.csv` and `.xlsx` are provided.

## Reference

The code is based on [MA](https://github.com/chaoma99/sr-metric), [NIQE](https://github.com/csjunxu/Bovik_NIQE_SPL2013), [PI](https://github.com/roimehrez/PIRM2018), [SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim) and [LPIPS](https://github.com/richzhang/PerceptualSimilarity). 
