import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np
import re


def cal_sim(tokenizer, model, query, block):
    # Tokenize input texts
    texts = [query, block.__str__()]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = tuple(t.cuda() for t in inputs.values())
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(*inputs, output_hidden_states=True, return_dict=True).pooler_output
    torch_sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    block.relevance = torch_sim
    torch_sim = torch_sim.cpu().item()
    return torch_sim


def cosine_similarity(text1, text2, tokenizer, model):
    texts = [text1, text2]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = tuple(t.cuda() for t in inputs.values())
    with torch.no_grad():
        # Get the embeddings
        embeddings = model(*inputs, output_hidden_states=True, return_dict=True).pooler_output
    torch_sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    torch_sim = round(torch_sim.cpu().item(),4)
    return torch_sim

def sim(context, tokenizer, model):
    sentences = re.split(r'(?:[.?!])', context)
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i < j:
                sim_mat[i][j] = cosine_similarity(sentences[i], sentences[j], tokenizer, model)
    return sim_mat
def all_sim_sentence(filename, name):
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda()
    data_frame = pd.read_csv(filename, sep='#', header=None)
    # txt_file_name = '{}.txt'.format(name)
    sim_list = []
    for index, row in tqdm(data_frame.iterrows()):
        sc_context = row[0]
        bc_pos_context = row[3]
        bc_context_candidates = list(row[5::2])
        sc_sim_list = sim(sc_context, tokenizer, model)
        answer = np.random.randint(0, len(bc_context_candidates) + 1)
        bc_pos_sim_list = sim(bc_pos_context,tokenizer,model)
        bc_sim_list = []
        for i in range(len(bc_context_candidates)):
            bc_neg_sim_list = sim(bc_context_candidates[i],tokenizer, model)
            bc_sim_list.append(bc_neg_sim_list)
        bc_sim_list.insert(answer, bc_pos_sim_list)
        sim_list.append(sc_sim_list)
        sim_list.append(bc_sim_list)
    np_sim = np.array(sim_list)
    np.savez('{}_sim_mat.npz'.format(name),np_sim)

# from hyperParams import BLOCK_SIZE, DEFAULT_MODEL_NAME
# Refer to https://github.com/lmh0921/keyB/blob/main/genePassage.py
class Block:
    # tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    # tokenModel =
    def __init__(self, tokens, pos, tokenizer):
        self.tokens = tokens
        self.pos = pos
        self.relevance = 0
        self.tokenizer = tokenizer

    def __str__(self):
        # tokids = self.tokenizer.convert_tokens_to_ids(self.tokens)
        # return self.tokenizer.decode(tokids)
        return self.tokenizer.convert_tokens_to_string(self.tokens)
    def __lt__(self, rhs):
        return self.pos < rhs.pos
    def __ne__(self, rhs):
        return self.pos != rhs.pos
    def __len__(self):
        return len(self.tokens)

class Passage:
    def __init__(self):
        self.blocks = []
    def insert(self, b):
        for index in range(len(self.blocks), -1, -1):
            if index == 0 or self.blocks[index - 1] < b:
                self.blocks.insert(index, b)
                break
    @staticmethod
    def split_document_into_blocks(d, tokenizer,BLOCK_SIZE, cnt=0, hard=True,):
        '''
            d: [['word', '##piece'], ...] # a document of tokenized sentences
            properties: [
                            [
                                (name: str, value: any), # len(2) tuple, sentence level property
                                (name: str, position: int, value: any) # len(3) tuple, token level property
                            ],
                            []... # len(d) lists
                        ]
        '''
        ret = Passage()
        updiv = lambda a, b: (a - 1) // b + 1
        if hard:
            for sid, tsen in enumerate(d):
                # psen = properties[sid] if properties is not None else []
                psen = []
                num = updiv(len(tsen), BLOCK_SIZE)  # cls
                bsize = updiv(len(tsen), num)
                for i in range(num):
                    st, en = i * bsize, min((i + 1) * bsize, len(tsen))
                    cnt += 1
                    # tmp = tsen[st: en] + [tokenizer.sep_token]
                    tmp = tsen[st: en]
                    # inject properties into blks
                    tmp_kwargs = {}
                    for p in psen:
                        if len(p) == 2:
                            tmp_kwargs[p[0]] = p[1]
                        elif len(p) == 3:
                            if st <= p[1] < en:
                                tmp_kwargs[p[0]] = (p[1] - st, p[2])
                        else:
                            raise ValueError('Invalid property {}'.format(p))
                    ret.insert(Block(tmp, cnt, tokenizer))
        else:
            # d is only a list of tokens, not split.
            # properties are also a list of tuples.
            end_tokens = {'\n': 0, '.': 1, '?': 1, '!': 1, ',': 2}
            # for k, v in list(end_tokens.items()):#BERT不需要
            #     end_tokens['Ġ' + k] = v
            sen_cost, break_cost = 4, 8
            poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if
                     tok in end_tokens]
            poses.insert(0, (-1, 0))
            if poses[-1][0] < len(d) - 1:
                poses.append((len(d) - 1, 0))
            x = 0
            while x < len(poses) - 1:
                if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                    poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
                x += 1
            # simple dynamic programming
            best = [(0, 0)]
            for i, (p, cost) in enumerate(poses):
                if i == 0:
                    continue
                best.append((-1, 100000))
                for j in range(i - 1, -1, -1):
                    if p - poses[j][0] > BLOCK_SIZE:
                        break
                    value = best[j][1] + cost + sen_cost
                    if value < best[i][1]:
                        best[i] = (j, value)
                assert best[i][0] >= 0

            intervals, x = [], len(poses) - 1
            while x > 0:
                l = poses[best[x][0]][0]
                intervals.append((l + 1, poses[x][0] + 1))
                x = best[x][0]
            for st, en in reversed(intervals):
                # copy from hard version
                cnt += 1
                # tmp = d[st: en] + [tokenizer.sep_token]
                tmp = d[st: en]
                ret.insert(Block(tmp, cnt, tokenizer))

        return ret, cnt
