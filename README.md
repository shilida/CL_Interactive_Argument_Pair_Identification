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
config/hyparameter.json is the config file. It contains a number of hyperparameters. Hyperparameters can be modified for custom training.  
noisy: `NO`,`RandomWordAug`,`BackTranslationAug`,`KeyboardAug`  
objective:`BCE`,`BCE+SCL`  
hard_sample_con:`NO`,`YES`  
model_type：`bert_without_context`,`bert_with_context`
## Test a model:
```shell
python test.py
```
## Contact
shild21@mails.jlu.edu.cn
## Citation
@inproceedings{shi2022simple, <br>
title={A Simple Contrastive Learning Framework for Interactive Argument Pair Identification via Argument-Context Extraction},<br>
author={Shi, Lida and Giunchiglia, Fausto and Song, Rui and Shi, Daqian and Liu, Tongtong and Diao, Xiaolei and Xu, Hao},<br>
booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},<br>
pages={10027--10039},<br>
year={2022}<br>
}<br>
