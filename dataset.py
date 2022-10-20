from torch.utils.data import Dataset
import torch
from torch.utils.data import TensorDataset
from transformers import LongformerTokenizer, BertTokenizer
from tqdm import tqdm
import pandas as pd
from ace import cal_sim,Passage
from transformers import AutoModel, AutoTokenizer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
class Data_WithContext:
    def __init__(self, config, max_seq_len, model_type):
        self.config = config
        self.model_type = model_type
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len
        self.context_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.context_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda()

    def load_train_and_dev_files(self, train_file, dev_file):
        print('Loading train records for train...')
        train_set = self.load_file(train_file, 'train')
        print(len(train_set), 'training records loaded.')
        print('Loading dev records...')
        dev_set = self.load_file(dev_file, 'dev')
        print(len(dev_set), 'dev records loaded.')
        return train_set,dev_set
    def load_valid_files(self,valid_file,noisy = False):
        print('Loading test records...')
        if noisy:
            valid_set = self.load_file(valid_file, 'valid',noisy=True)
        else:
            valid_set = self.load_file(valid_file, 'valid')
        print(len(valid_set), 'valid records loaded.')
        return valid_set

    def load_file(self,file_path, name='train') -> TensorDataset:
        quo_list, reply_list, label_list = self.loaddata(file_path,name,noisy=False)
        dataset = self._convert_sentence_pair_to_bert(quo_list, reply_list, label_list)
        return dataset
    def loaddata(self, filename, name,noisy=False):
        data_frame = pd.read_csv(filename, sep='#', header=None)
        quo_list, reply_list, label_list = [], [], []
        for index, row in tqdm(data_frame.iterrows()):
            quo_context = row[0]
            quo = row[1]
            reply_pos = row[2]
            reply_pos_context = row[3]
            reply_neg = list(row[4::2])
            reply_context_candidates = list(row[5::2])  # 取出从第三个数开始的偶数项
            if noisy==False:
                quo_partcontext = self._selectblock(quo, quo_context)
            else:
                quo_partcontext = self._selectblock_noisy(quo, quo_context)
            reply_partcontext_list = []
            # 1个quo1个reply_pos 4个reply_neg
            if noisy == False:
                reply_pos_partcontext = self._selectblock(reply_pos, reply_pos_context)
            else:
                reply_pos_partcontext = self._selectblock_noisy(reply_pos, reply_pos_context) # self._selectpartcontext_sim(reply_pos, reply_pos_context,reply_sim_pos)  # random.randint(0,len(reply_context_candidates))  # random.randint(0, len(candidates))  # 随机生成label位置
            reply_partcontext_list.append(reply_pos_partcontext)
            label_list.append(1)
            for i in range(len(reply_context_candidates)):
                if noisy == False:
                    reply_neg_partcontext = self._selectblock(reply_neg[i], reply_context_candidates[i])
                else:
                    reply_neg_partcontext = self._selectblock_noisy(reply_neg[i], reply_context_candidates[i])
                # reply_neg_partcontext = self._selectblock(reply_neg[i], reply_context_candidates[i])# self._selectpartcontext_sim(reply_neg[i], reply_context_candidates[i], reply_sim[i])
                reply_partcontext_list.append(reply_neg_partcontext)
                label_list.append(0)
            if name == 'train':
                # pos_reply:neg_reply = 1:2
                pos_reply_contrast_1 = self._selectblock_pos(reply_pos, reply_pos_context,self.config.pos1_block_size,self.config.pos1_block_num)
                pos_reply_contrast_2 = self._selectblock_pos(reply_pos, reply_pos_context,self.config.pos2_block_size,self.config.pos2_block_num)
                pos_reply_contrast_3 = self._selectblock_pos(reply_pos, reply_pos_context, self.config.pos3_block_size,self.config.pos3_block_num)
                reply_partcontext_list.append(pos_reply_contrast_1)
                reply_partcontext_list.append(pos_reply_contrast_2)
                reply_partcontext_list.append(pos_reply_contrast_3)
                label_list.append(1)
                label_list.append(1)
                label_list.append(1)
                neg_reply_contrast_1 = self._selectblock_neg(reply_neg[0], reply_context_candidates[0],self.config.pos1_block_size,self.config.pos1_block_num)  # self._selectblock_pos(reply_pos, reply_pos_context,self.config.pos1_block_size,self.config.pos1_block_num)
                neg_reply_contrast_2 = self._selectblock_neg(reply_neg[1], reply_context_candidates[1],self.config.pos2_block_size,self.config.pos2_block_num)  # self._selectblock_pos(reply_pos, reply_pos_context,self.config.pos2_block_size,self.config.pos2_block_num)
                neg_reply_contrast_3 = self._selectblock_neg(reply_neg[2], reply_context_candidates[2],self.config.pos3_block_size,self.config.pos3_block_num)  # self._selectblock_pos(reply_pos, reply_pos_context, self.config.pos3_block_size,self.config.pos3_block_num)
                neg_reply_contrast_4 = self._selectblock_neg(reply_neg[3], reply_context_candidates[3], self.config.block_num,self.config.block_size)
                reply_partcontext_list.append(neg_reply_contrast_1)
                reply_partcontext_list.append(neg_reply_contrast_2)
                reply_partcontext_list.append(neg_reply_contrast_3)
                reply_partcontext_list.append(neg_reply_contrast_4)
                label_list.append(0)
                label_list.append(0)
                label_list.append(0)
                label_list.append(0)
            quo_list.append(reply_partcontext_list)
            reply_list.append(quo_partcontext)
        return quo_list, reply_list, label_list
    def _selectblock_noisy(self, query, context):
        passage, cnt = Passage.split_document_into_blocks(self.tokenizer.tokenize(context), self.tokenizer,
                                                          self.config.block_size, 0, hard=False)
        sim_list = []
        for con in passage.blocks:
            sim_list.append(cal_sim(self.context_tokenizer, self.context_model, query, con))
        sorted_id = sorted(range(len(sim_list)), key=lambda k: sim_list[k], reverse=True)
        newindex = sorted_id[:self.config.block_num]  #
        sorted_new_index = sorted(newindex)
        part_list = []
        part_list.append(query)
        for i in sorted_new_index:
            part_list.append(passage.blocks[i].__str__())
        con_str = ""
        for part_str in part_list:
            con_str = con_str + part_str
        # different noisy
        aug_random = naw.RandomWordAug()
        augmented_text = aug_random.augment(con_str)
        # aug_backtranlation = naw.BackTranslationAug()
        # augmented_text = aug_backtranlation.augment(augmented_text)
        # print('augmented_text_BackTranslationAug', augmented_text)
        # aug_key = nac.KeyboardAug()
        # augmented_text = aug_key.augment(augmented_text)
        # print('augmented_text_KeyboardAug', augmented_text)
        # aug_backtranlation = naw.BackTranslationAug()
        # augmented_text = aug_backtranlation.augment(augmented_text)
        # aug_key = nac.KeyboardAug()
        # augmented_text = aug_key.augment(con_str)
        return augmented_text
    def _selectblock(self, query, context):
        passage, cnt = Passage.split_document_into_blocks(self.tokenizer.tokenize(context), self.tokenizer,
                                                          self.config.block_size, 0, hard=False)
        sim_list = []
        for con in passage.blocks:
            sim_list.append(cal_sim(self.context_tokenizer, self.context_model, query, con))
        sorted_id = sorted(range(len(sim_list)), key=lambda k: sim_list[k], reverse=True)
        # random.shuffle(sorted_id)#不随机的时候注释掉
        newindex = sorted_id[:self.config.block_num]
        sorted_new_index = sorted(newindex)
        part_list = []
        part_list.append(query)
        for i in sorted_new_index:
            part_list.append(passage.blocks[i].__str__())
        con_str = ""
        for part_str in part_list:
            con_str = con_str + part_str
        return con_str
    def _selectblock_pos(self, query, context,pos_block_size,pos_block_num):
        passage, cnt = Passage.split_document_into_blocks(self.tokenizer.tokenize(context), self.tokenizer,
                                                          pos_block_size, 0, hard=False)
        sim_list = []
        for con in passage.blocks:
            sim_list.append(cal_sim(self.context_tokenizer, self.context_model, query, con))
        sorted_id = sorted(range(len(sim_list)), key=lambda k: sim_list[k], reverse=True)
        # random.shuffle(sorted_id)  # 不随机的时候注释掉
        newindex = sorted_id[:pos_block_num]  #
        sorted_new_index = sorted(newindex)
        part_list = []
        part_list.append(query)
        for i in sorted_new_index:
            part_list.append(passage.blocks[i].__str__())
        con_str = ""
        for part_str in part_list:
            con_str = con_str + part_str
        return con_str
    def _selectblock_neg(self, query, context,pos_block_size,pos_block_num):
        passage, cnt = Passage.split_document_into_blocks(self.tokenizer.tokenize(context), self.tokenizer,
                                                          pos_block_size, 0, hard=False)
        sim_list = []
        for con in passage.blocks:
            sim_list.append(cal_sim(self.context_tokenizer, self.context_model, query, con))
        sorted_id = sorted(range(len(sim_list)), key=lambda k: sim_list[k], reverse=True)
        # random.shuffle(sorted_id)
        newindex = sorted_id[:pos_block_num]
        sorted_new_index = sorted(newindex)
        part_list = []
        part_list.append(query)
        for i in sorted_new_index:
            part_list.append(passage.blocks[i].__str__())
        con_str = ""
        for part_str in part_list:
            con_str = con_str + part_str
        return con_str
    def _convert_sentence_pair_to_bert(self, quo_list, reply_list, label_list=None):
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(quo_list), ncols=80):
            token_list = []
            mask_list = []
            segment_list = []
            quo_list[i] = self.tokenizer.tokenize(quo_list[i])
            for j, _ in enumerate(reply_list[i]):
                reply_list[i][j] = self.tokenizer.tokenize(reply_list[i][j])
                tokens = ['[CLS]'] + quo_list[i] + ['[SEP]']
                segment_ids = [0] * len(tokens)
                tokens += reply_list[i][j] + ['[SEP]']
                segment_ids += [1] * (len(reply_list[i][j]) + 1)
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                    segment_ids = segment_ids[:self.max_seq_len]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                tokens_len = len(input_ids)
                input_ids += [0] * (self.max_seq_len - tokens_len)
                segment_ids += [0] * (self.max_seq_len - tokens_len)
                input_mask += [0] * (self.max_seq_len - tokens_len)
                token_list.append(input_ids)
                mask_list.append(input_mask)
                segment_list.append(segment_ids)
            all_input_ids.extend(token_list)
            all_input_mask.extend(mask_list)
            all_segment_ids.extend(segment_list)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        # if label_list:  # train
        all_label_ids = torch.tensor(label_list, dtype=torch.long)
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
class Data_WithoutContext:
    def __init__(self,
                 config,
                 max_seq_len: int = 512,
                 model_type: str = 'bert'):
        self.config = config
        self.model_type = model_type
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len

    def load_train_and_dev_files(self, train_file, dev_file):
        print('Loading train records for train...')
        train_set = self.load_file(train_file)
        print(len(train_set), 'training records loaded.')
        print('Loading dev records...')
        dev_set = self.load_file(dev_file)
        print(len(dev_set), 'dev records loaded.')
        return train_set, dev_set

    def load_valid_files(self, valid_file):
        print('Loading valid records...')
        valid_set = self.load_file(valid_file)
        print(len(valid_set), 'valid records loaded.')
        return valid_set

    def load_file(self,file_path='train.txt') -> TensorDataset:
        quo_list, reply_list, label_list = self._load_file(file_path)
        dataset = self._convert_sentence_pair_to_bert(
            quo_list, reply_list, label_list)
        return dataset
    def _load_file(self, filename):
        data_frame = pd.read_csv(filename, sep='#', header=None)[:500]
        quo_list, reply_list, label_list = [], [], []
        for index, row in data_frame.iterrows():
            candidates = list(row[4::2])
            quo_tokens = row[1]
            reply_label = row[2]
            reply_list_every_row = []
            reply_list_every_row.append(reply_label)
            label_list.append(1)
            for i in range(len(candidates)):
                reply_tokens_no_label = candidates[i]
                reply_list_every_row.append(reply_tokens_no_label)
                label_list.append(0)
            reply_list.append(reply_list_every_row)
            quo_list.append(quo_tokens)
        return quo_list, reply_list, label_list
    def _convert_sentence_pair_to_bert(self, quo_list, reply_list, label_list=None, ):
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(quo_list), ncols=80):
            token_list = []
            mask_list = []
            segment_list = []
            quo_list[i] = self.tokenizer.tokenize(quo_list[i])
            for j, _ in enumerate(reply_list[i]):
                reply_list[i][j] = self.tokenizer.tokenize(reply_list[i][j])
                tokens = ['[CLS]'] + quo_list[i] + ['[SEP]']
                segment_ids = [0] * len(tokens)
                tokens += reply_list[i][j] + ['[SEP]']
                # segment_ids += [j+1] * (len(s2_list[i][j]) + 1)
                segment_ids += [1] * (len(reply_list[i][j]) + 1)
                if len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]
                    segment_ids = segment_ids[:self.max_seq_len]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                tokens_len = len(input_ids)
                input_ids += [0] * (self.max_seq_len - tokens_len)
                segment_ids += [0] * (self.max_seq_len - tokens_len)
                input_mask += [0] * (self.max_seq_len - tokens_len)
                token_list.append(input_ids)
                mask_list.append(input_mask)
                segment_list.append(segment_ids)
            all_input_ids.extend(token_list)
            all_input_mask.extend(mask_list)
            all_segment_ids.extend(segment_list)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        # if label_list:  # train
        all_label_ids = torch.tensor(label_list, dtype=torch.long)
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)