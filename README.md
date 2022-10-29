# A Simple Contrastive Learning Framework for Interactive Argument Pair Identification via Argument-Context Extraction(EMNLP 2022)

The method is implemented using PyTorch.
## Folder
- `config`: the config and hyperparameter file
- `data`: please put data in this folder
- `dataset.py`: data preprocessing file
- `models.py`: model file.
- `evaluation.py`: evaluation function file
- `train.py`: training, validation function file 
- `test.py`: testing function
- `ace.py`:  argument-context extraction module
## Train a model:
```shell
python train.py
```
config/hyparameter.json is the config files. It contains a number of hyperparameters. Hyperparameters can be modified for custom training.  
noisy: `NO`,`RandomWordAug`,`BackTranslationAug`,`KeyboardAug`  
objective:`BCE`,`BCE+SCL`
## Test a model:
```shell
python test.py
```
## Citation
