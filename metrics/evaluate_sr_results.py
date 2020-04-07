import os
import cv2
import math
import logging
import datetime
import pandas as pd
from PIL import Image
import LPIPS as models
import matlab.engine
import torch
import argparse
from tqdm import tqdm
from logging import handlers
import numpy as np
import yaml

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
        }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

def CalMATLAB(SRFolder,GTFolder):
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(eng.fullfile(os.getcwd(),'MetricEvaluation')))
    res = eng.evaluate_results(SRFolder,GTFolder)
    res=np.array(res)
    res=res.squeeze()
    return res

def CalLPIPS(SRFolder,GTFolder):
    nameList=os.listdir(SRFolder)
    res=[]
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=False)
    for i in nameList:
        imageA=os.path.join(SRFolder,i)
        imageB=os.path.join(GTFolder,i)
        imageA = np.array(Image.open(imageA))
        imageB = np.array(Image.open(imageB))
        imageA = torch.Tensor((imageA / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
        imageB = torch.Tensor((imageB / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
        dist = model.forward(imageA,imageB).detach().squeeze().numpy()
        res.append(dist)
    res=np.array(res)
    res=res.squeeze()
    return np.mean(res)

parser = argparse.ArgumentParser(description="Evaluate SR results")
parser.add_argument('YAML', type=str, help='configuration file')
args = parser.parse_args()

conf=dict()
with open(args.YAML,'r', encoding='UTF-8') as f:
    conf=yaml.load(f.read())

Datasets=conf['Pairs']['Dataset']
SRFolder=conf['Pairs']['SRFolder']
GTFolder=conf['Pairs']['GTFolder']
Metric=['Ma','NIQE','PI','PSNR','SSIM','MSE','RMSE','LPIPS']
Name=conf['Name']
Echo=conf['Echo']

output=Name+datetime.datetime.now().strftime('-%Y%m%d%H%M%S')
if not os.path.isdir('../evaluate'):
    os.mkdir('../evaluate')

os.mkdir(os.path.join('../evaluate',output))

log = Logger(os.path.join('../evaluate',output+'.log'),level='info')

log.logger.info('Init...')
log.logger.info('SRFolder - '+str(Datasets))
log.logger.info('GTFolder - '+str(GTFolder))
log.logger.info('SRFolder - '+str(SRFolder))
log.logger.info('Metric - '+str(Metric))
log.logger.info('Name - '+Name)
log.logger.info('Echo - '+str(Echo))

res = pd.DataFrame(columns=('PI','Ma','NIQE','MSE','RMSE','PSNR','SSIM','LPIPS'))

for i,j,k in zip(Datasets,SRFolder,GTFolder):
    log.logger.info('Calculating '+i+'...')
    assert set(os.listdir(j))==set(os.listdir(k)), 'SR pictures and GT pictures are not matched.'
    MATLAB=CalMATLAB(j,k)
    LPIPS=CalLPIPS(j,k)
    resDict=dict()
    resDict['PI']=[MATLAB[0]]
    resDict['Ma']=[MATLAB[1]]
    resDict['NIQE']=[MATLAB[2]]
    resDict['MSE']=[MATLAB[3]]
    resDict['RMSE']=[MATLAB[4]]
    resDict['PSNR']=[MATLAB[5]]
    resDict['SSIM']=[MATLAB[6]]
    resDict['LPIPS']=[LPIPS]
    resDataFrame=pd.DataFrame(resDict)
    resDataFrame.index=[i]
    res=res.append(resDataFrame)
    if Echo:
        log.logger.info('['+i+']    PI - '+str(MATLAB[0]))
        log.logger.info('['+i+']    Ma - '+str(MATLAB[1]))
        log.logger.info('['+i+']  NIQE - '+str(MATLAB[2]))
        log.logger.info('['+i+']   MSE - '+str(MATLAB[3]))
        log.logger.info('['+i+']  RMSE - '+str(MATLAB[4]))
        log.logger.info('['+i+']  PSNR - '+str(MATLAB[5]))
        log.logger.info('['+i+']  SSIM - '+str(MATLAB[6]))
        log.logger.info('['+i+'] LPIPS - '+str(LPIPS))

res.to_csv(os.path.join('../evaluate',output,Name+'.csv'), header=True, index=True)
res.to_excel(os.path.join('../evaluate',output,Name+'.xlsx'), header=True, index=True)

log.logger.info('Done.')
