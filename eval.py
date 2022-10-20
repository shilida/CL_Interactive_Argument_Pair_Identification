import numpy as np
from typing import List
import torch
from tqdm import tqdm
from sklearn import metrics
import codecs
import pandas as pd

def compute_acc(label: list, logits: list) -> float:
    # Calculate the Accuracy according to the given output logits.
    count = 0
    for i in range(len(label)):
        if label[i] == np.array(logits[i]).argmax():
            count += 1
    return count / len(label)


def compute_mrr(label: list, logits: list) -> float:
    # Calculate the MRR according to the given output logits.
    count = 0
    for i in range(len(label)):
        order = np.array(logits[i]).argsort()[::-1]
        for j in range(len(order)):
            if order[j] == label[i]:
                count += 1.0 / (j + 1)
    return count / len(label)
def evaluate_model(model, data_loader, device) -> List[str]:
    model.eval()
    outputs = torch.tensor([], dtype=torch.float).to(device)
    label = []
    for batch in tqdm(data_loader, desc='Evaluation', ncols=80):
        batch = tuple(t.to(device) for t in batch)
        # print(batch)
        with torch.no_grad():
            logits = model(*batch[:-1],train = False)
        size = logits.shape[0]
        batch_label = torch.zeros(size)
        label.extend(batch_label)
        outputs = torch.cat([outputs, logits])
    outputs = outputs.cpu()
    return outputs, label
