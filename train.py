import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from typing import Dict
import torch
import argparse
from tqdm import tqdm, trange
from copy import deepcopy
from types import SimpleNamespace
from models import BertFor2Classification
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import eval
from dataset import Data_WithContext,Data_WithoutContext
import os
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
class Trainer:
    def __init__(self,
                 model, data_loader: Dict[str, DataLoader], device, config, distributed):
        self.model = model
        self.device = device
        self.config = config
        self.data_loader = data_loader
        self.num_training_steps = config.num_epoch * len(data_loader['train'])
        self.awl = AutomaticWeightedLoss(2).cuda()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.CrossEntropyLoss()
        self.distributed = distributed

    def _get_optimizer(self):
        """Get optimizer for different models.

        Returns:
            optimizer
        """
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0},
            {'params': self.awl.parameters(), 'weight_decay': 0}
        ]
        optimizer = AdamW(
            optimizer_parameters,
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-8,
            correct_bias=False)
        return optimizer

    def _get_scheduler(self):
        """Get scheduler for different models.
        Returns:
            scheduler
        """
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps * self.num_training_steps,
            num_training_steps=self.num_training_steps)
        return scheduler
    def _epoch_evaluate(self):
        dev_predictions, label_dev = eval.evaluate_model(
            model=self.model, data_loader=self.data_loader['dev'],
            device=self.device)
        dev_acc = eval.compute_acc(label=label_dev, logits=dev_predictions)
        dev_mrr = eval.compute_mrr(label=label_dev, logits=dev_predictions)
        return dev_acc,dev_mrr
    def save_model(self, filename):
        """Save model to file.

        Args:
            filename: file name
        """
        torch.save(self.model.state_dict(), filename)
    def sim_matrix(self, a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def contrastive(self, embedding, label, temp):
        """calculate the contrastive loss
        """
        # cosine similarity between embeddings
        cosine_sim = self.sim_matrix(embedding, embedding)
        n = cosine_sim.shape[0]
        a = ~torch.eye(n, dtype=bool).cuda()
        dis = cosine_sim.masked_select(a).view(n, n - 1)

        # apply temperature to elements
        dis = dis / temp
        cosine_sim = cosine_sim / temp
        # apply exp to elements
        dis = torch.exp(dis)
        cosine_sim = torch.exp(cosine_sim)

        # calculate row sum
        row_sum = torch.sum(dis, -1)

        unique_labels, inverse_indices, unique_label_counts = torch.unique(label, sorted=False, return_inverse=True,
                                                                           return_counts=True)
        # calculate outer sum
        contrastive_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        for i in range(n):
            n_i = unique_label_counts[inverse_indices[i]] - 1
            inner_sum = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
            # calculate inner sum
            for j in range(n):
                if label[i] == label[j] and i != j:
                    inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
            if n_i != 0:
                contrastive_loss += (inner_sum / (-n_i))
        return contrastive_loss
    def cal_contrastive_loss(self, emb, label, temp):
        num = emb.shape[0]
        loss = self.contrastive(emb.cuda(), label.cuda(), torch.tensor(temp).cuda())
        loss_mean = loss / num
        return loss_mean
    def train(self):
        scaler = GradScaler()
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[0, 1, 2, 3], find_unused_parameters=True)
        # wandb.watch(self.model)
        trange_obj = trange(self.config.num_epoch, desc='Epoch', ncols=120)
        best_model_state_dict, best_dev_acc, global_step = None, 0, 0
        for epoch, _ in enumerate(trange_obj):
            self.model.train()
            tqdm_obj = tqdm(self.data_loader['train'], ncols=80)
            count = 0
            loss_avg = 0
            for step, batch in enumerate(tqdm_obj):
                self.optimizer.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                label = batch[-1]
                with autocast():
                    logits, emb = self.model(*batch[:-1], train=True)  # the last one is label
                    loss_classifer = self.criterion(logits.squeeze(),
                                                    label.float())  # self.convert_label_to_onehot(label).float()
                    if self.config.objective == 'BCE':
                        loss = loss_classifer
                    if self.config.objective == 'BCE+SCL':
                        loss_contrastive = self.cal_contrastive_loss(emb, label, self.config.temp)
                        loss = self.awl(loss_classifer,loss_contrastive)
                count = count + 1
                loss_avg = loss_avg + loss
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                if count % 10 == 0:
                    loss_avg = loss_avg / 10
                    # wandb.log({
                    #     "loss" + self.config.model_type: loss_avg,
                    #     "lr" + self.config.model_type: lr
                    # })
                    loss_avg = 0
                    count = 0
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    global_step += 1
                    tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))
            dev_acc,dev_mrr = self._epoch_evaluate()
            print('epoch_{}:dev_acc_{},dev_mrr_{}'.format(epoch,dev_acc,dev_mrr))
            if dev_acc > best_dev_acc:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_dev_acc = dev_acc
        print("best_dev_acc:",best_dev_acc)
        return best_model_state_dict,best_dev_acc
def _init_fn(worker_id):
    np.random.seed(int(0))
def get_path(path):
    """Create the path if it does not exist.
    Args:
        path: path to be used
    Returns:
        Existed path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def main_fun(config_file='config/hyparam.json'):
    """Main method for training.
    Args:
        distributed: if distributed train.
    """
    # 0. Load config and mkdir
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_save_path, config.model_type))
    if args.block_num!=0:
        config.block_num = args.block_num
    if args.block_size != 0:
        config.block_size = args.block_size
    if args.temp != 0:
        config.temp = args.temp
    # wandb.config.update(config)
    # 1. Load data
    print(">>>>>load data")
    if config.model_type == 'bert_without_context':
        data = Data_WithoutContext(config=config,
                              max_seq_len=config.max_seq_len,
                              model_type=config.model_type)
    if config.model_type == 'bert_with_context':
        data = Data_WithContext(config=config,
                                          max_seq_len=config.max_seq_len,
                                          model_type=config.model_type)  # Data_ContextMultipleChoice
    if config.noisy == 'NO':
        noisy = False
    else:
        noisy = config.noisy
    if config.hard_sample_con == 'NO':
        hard_sample_con = False
    else:
        hard_sample_con = True
    train_set, dev_set = data.load_train_and_dev_files(
        train_file=config.train_file_path,
        dev_file=config.dev_file_path,hard_sample_con = hard_sample_con,noisy=noisy)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if args.distributed:
            print("Using distributed for train!!")
            torch.distributed.init_process_group(backend="nccl")
            sampler_train = DistributedSampler(train_set, shuffle=True, seed=seed)
        else:
            sampler_train = RandomSampler(train_set)
    else:
        device = torch.device('cpu')
        sampler_train = RandomSampler(train_set)
    data_loader = {
        'train': DataLoader(
            train_set, sampler=sampler_train, batch_size=config.batch_size, num_workers=16, worker_init_fn=_init_fn),
        'dev': DataLoader(
            dev_set, batch_size=config.batch_size, shuffle=False, num_workers=16, worker_init_fn=_init_fn)}
    # 2. Build model
    print(">>>>>Build model")
    model = BertFor2Classification(config)
    model.to(device)
    # 3. Train and Save model
    print(">>>>>Train and Save model")
    trainer = Trainer(model=model, data_loader=data_loader,
                      device=device, config=config, distributed=args.distributed)
    best_model_dev_dict,best_dev_acc = trainer.train()
    # 4. Save model
    print(">>>>>Save model")
    torch.save(best_model_dev_dict,
               os.path.join(config.model_save_path, 'model_{}.bin'.format(best_dev_acc)))
if __name__ == '__main__':
    # torch.cuda.set_device(1)
    parser = argparse.ArgumentParser()
    # You can also use the parser to adjust hyparameters
    parser.add_argument('--local_rank', default=0, help='used for distributed parallel')
    parser.add_argument("--distributed", action="store_true", help="if distributed train.")
    parser.add_argument("--block_num", type=int, default=0, help="block num")
    parser.add_argument("--block_size", type=int, default=0, help="block size")
    parser.add_argument("--temp", type=float, default=0, help="block size")
    args = parser.parse_args()
    main_fun()
