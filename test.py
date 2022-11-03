
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import random
from dataset import Data_WithoutContext,Data_WithContext
from models import  BertFor2Classification
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def compute_acc(label: list, logits: list):
    # Calculate the Accuracy according to the given output logits.
    count = 0
    for i in range(len(label)):
        if label[i] == np.array(logits[i]).argmax():
            count += 1
    return count / len(label)
def compute_mrr(label: list, logits: list):
    # Calculate the MRR according to the given output logits.
    count = 0
    for i in range(len(label)):
        order = np.array(logits[i]).argsort()[::-1]
        for j in range(len(order)):
            if order[j] == label[i]:
                count += 1.0 / (j + 1)
    return count / len(label)
def evaluate(model, data_loader, device):
    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    label_acc = []
    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch)
        # print(batch)
        with torch.no_grad():
            logits, feature = model(*batch[:-1], train=True)
        logits= logits.view(-1, 5)
        batch_label_acc = batch[-1].view(-1, 5).argmax(axis=1) #torch.zeros(size)
        label_acc.extend(batch_label_acc)
        outputs = torch.cat([outputs, logits])
    outputs = outputs.cpu()
    return outputs, label_acc
def _init_fn(worker_id):
    np.random.seed(int(0))
def test_model(model,data_loader):
    valid_predictions, label_valid = evaluate(
        model=model, data_loader=data_loader['valid'],
        device='cuda')
    test_acc = compute_acc(label=label_valid, logits=valid_predictions)
    test_mrr = compute_mrr(label=label_valid, logits=valid_predictions)
    print("test_acc",test_acc)
    print("test_mrr",test_mrr)
def testt():
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    config_file = 'config/hyparam.json'
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    assert config.eval_batch_size % 5 == 0, "eval_batch_size must be divisible by 5"
    model1 = BertFor2Classification(config)
    model1.cuda()
    state_dict1 = torch.load('model/with_context/model_0.8133102852203976.bin')  # model_0.8193604148660328.bin
    model1.load_state_dict(state_dict1)
    data = Data_WithContext(config=config,max_seq_len=config.max_seq_len,model_type=config.model_type)
    if config.noisy == 'NO':
        noisy = False
    else:
        noisy = config.noisy
    valid_set = data.load_valid_files(config.test_file_path,noisy)
    data_loader = {
        'valid': DataLoader(
            valid_set, batch_size=config.eval_batch_size, shuffle=False, num_workers=8, worker_init_fn=_init_fn)}
    test_model(model1, data_loader)
if __name__ == '__main__':
    testt()
