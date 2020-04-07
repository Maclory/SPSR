import os
import yaml
import numpy as np

def dict2Yaml(fileName, dataDict):
    stream = open(fileName, 'w', encoding='UTF-8')
    yaml.dump(dataDict, stream=stream, default_flow_style=False)
    conf=dict()

MethodDict=dict()
MethodDict['SPSR_0330']=['Set5', 'Set14', 'BSD100', 'Urban100']

if not os.path.isdir('Configuration'):
    os.mkdir('Configuration')

bashFile=[]

for method in MethodDict.keys():
    fileName = method+'.yml'
    dataDict=dict()
    dataDict['Pairs']=dict()
    dataDict['Pairs']['Dataset']=MethodDict[method]
    dataDict['Pairs']['SRFolder']=[]
    dataDict['Pairs']['GTFolder']=[]
    dataDict['Name']=fileName
    dataDict['Echo']=True
    for dataset in MethodDict[method]:
        dataDict['Pairs']['SRFolder'].append(str(os.path.join('../data',method,dataset)))
        dataDict['Pairs']['GTFolder'].append(str(os.path.join('../data','GT',dataset,'HR')))
    bashFile.append('python evaluate_sr_results.py '+str(os.path.join('Configuration',fileName)))
    dict2Yaml(os.path.join('Configuration',fileName), dataDict)

np.savetxt('Run.bash',np.array(bashFile),fmt='%s')

print('Done.')
