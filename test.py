import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
import json
import os
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
    outputs_emb = torch.tensor([], dtype=torch.float).to(device)
    label_acc = []
    label_tsne = []
    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch)
        # print(batch)
        with torch.no_grad():
            logits, feature = model(*batch[:-1], train=True)
        logits= logits.view(-1, 5)
        size = logits.shape[0]
        batch_label_acc = torch.zeros(size)
        # loss = criterion(logits, batch[-1])
        label_acc.extend(batch_label_acc)
        label_tsne.extend(batch[-1])
        outputs = torch.cat([outputs, logits])
        outputs_emb = torch.cat([outputs_emb, feature])
    outputs = outputs.cpu()
    outputs_emb = outputs_emb.cpu()
    return outputs, label_acc, outputs_emb,label_tsne
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def tsne(model1,model2, data_loader):
    valid_predictions_cl, label_valid_cl, outputs_emb_valid_cl,label_tsne_valid_cl = evaluate(
        model=model1, data_loader=data_loader['valid'],
        device='cuda')
    valid_predictions_no_cl, label_valid_no_cl, outputs_emb_valid_no_cl,label_tsne_valid_no_cl = evaluate(
        model=model2, data_loader=data_loader['valid'],
        device='cuda')
    # result = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(outputs_emb_valid)
    result_cl = TSNE(n_components=2, random_state=42).fit_transform(outputs_emb_valid_cl)
    result_no_cl = TSNE(n_components=2, random_state=42).fit_transform(outputs_emb_valid_no_cl)
    # Create the figure
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(result_no_cl[:, 0], result_no_cl[:, 1], c=label_tsne_valid_no_cl)#, label="CE"
    plt.legend()
    plt.subplot(122)
    plt.scatter(result_cl[:, 0], result_cl[:, 1], c=label_tsne_valid_cl)# label="CE+CL"
    plt.legend()
    plt.show()
    plt.savefig('vis/tsne.png', dpi=120)
    # valid_acc = compute_acc(label=label_valid, logits=valid_predictions)
    # valid_mrr = compute_mrr(label=label_valid, logits=valid_predictions)
    # dev_acc = compute_acc(label=label_dev, logits=dev_predictions)
    # dev_mrr = compute_mrr(label=label_dev, logits=dev_predictions)
    # return valid_acc, valid_mrr, dev_acc, dev_mrr  # train_acc, train_mrr, train_loss,
def _init_fn(worker_id):
    np.random.seed(int(0))
def test_model(model,data_loader):
    valid_predictions_cl, label_valid_cl, outputs_emb_valid_cl, label_tsne_valid_cl = evaluate(
        model=model, data_loader=data_loader['valid'],
        device='cuda')
    test_acc = compute_acc(label=label_valid_cl, logits=valid_predictions_cl)
    test_mrr = compute_mrr(label=label_valid_cl, logits=valid_predictions_cl)
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
    config_file = 'config/hyparameter.json'
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    model1 = BertFor2Classification(config)
    model1.cuda()
    state_dict1 = torch.load('model/with_context/model_0.8193604148660328.bin')  # model_0.8193604148660328.bin
    model1.load_state_dict(state_dict1)
    # model2 = BertFor2Classification(config)
    # model2.cuda()
    # state_dict2 = torch.load('model/with_context/model_0.7535449020931803.bin')
    # model2.load_state_dict(state_dict2)
    data = Data_WithContext(config=config,max_seq_len=config.max_seq_len,model_type=config.model_type)
    if config.noisy == 'NO':
        noisy = False
    else:
        noisy = config.noisy
    valid_set = data.load_valid_files(config.test_file_path,noisy)
    data_loader = {
        'valid': DataLoader(
            valid_set, batch_size=config.batch_size, shuffle=False, num_workers=8, worker_init_fn=_init_fn)}
    test_model(model1, data_loader)
if __name__ == '__main__':
    testt()
