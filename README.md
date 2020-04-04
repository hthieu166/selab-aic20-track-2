# selab-aic20-track-2

### Installation

Using conda environment. Environment file: 
```
environment.yaml
```
Download track 2 dataset, config folder directory under
```
configs/dataset_cfgs
```
Track2 data on selab3 server:
```
/home/hthieu/aicity2020/aic20_data/origin
```
For each experiment, training and model configurations should be defined in
```
configs/model_cfgs
configs/train_cfgs
```
New class (model/loss/data augmentation) should be implemented in corresponding folder and registered under 
```
factories.py
```