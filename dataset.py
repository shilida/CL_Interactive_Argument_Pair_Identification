from torch.utils.data import Dataset
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer,BertModel
from tqdm import tqdm
import pandas as pd
from ace import cal_sim, Passage
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac


class Data_WithContext:
    def __init__(self, config, max_seq_len, model_type):
        self.config = config
        self.model_type = model_type
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len
        self.context_tokenizer = BertTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.context_model = BertModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda()
    def load_train_and_dev_files(self, train_file, dev_file, hard_sample_con=False, noisy=False):
        print('Loading train data...')
        train_set = self.load_file(train_file, hard_sample_con, noisy)
        print(len(train_set), 'train data loaded.')
        print('Loading dev data...')
        dev_set = self.load_file(dev_file, False, noisy)
        print(len(dev_set), 'dev data loaded.')
        return train_set, dev_set
    def load_valid_files(self, valid_file, noisy=False):
        print('Loading test data...')
        valid_set = self.load_file(valid_file, False, noisy)
        print(len(valid_set), 'test data loaded.')
        return valid_set
    def load_file(self, file_path, hard_construction=False, noisy=False) -> TensorDataset:
        quo_list, reply_list, label_list = self.loaddata(file_path, hard_construction, noisy=noisy)
        dataset = self._convert_sentence_pair_to_bert(quo_list, reply_list, label_list)
        return dataset
    def loaddata(self, filename, hard_construction, noisy=False):
        data_frame = pd.read_csv(filename, sep='#', header=None)
        quo_list, reply_list, label_list = [], [], []
        for index, row in tqdm(data_frame.iterrows()):
            quo_context = row[0]
            quo = row[1]
            reply_pos = row[2]
            reply_pos_context = row[3]
            reply_neg = list(row[4::2])
            reply_context_candidates = list(row[5::2])  # 取出从第三个数开始的偶数项
            quo_partcontext = self._selectblock(quo, quo_context, noisy)
            reply_partcontext_list = []
            # 1个quo1个reply_pos 4个reply_neg
            reply_pos_partcontext = self._selectblock(reply_pos, reply_pos_context, noisy)
            # self._selectpartcontext_sim(reply_pos, reply_pos_context,reply_sim_pos)  # random.randint(0,len(reply_context_candidates))  # random.randint(0, len(candidates))  # 随机生成label位置
            reply_partcontext_list.append(reply_pos_partcontext)
            label_list.append(1)
            for i in range(len(reply_context_candidates)):
                reply_neg_partcontext = self._selectblock(reply_neg[i], reply_context_candidates[i], noisy)
                # reply_neg_partcontext = self._selectblock(reply_neg[i], reply_context_candidates[i])# self._selectpartcontext_sim(reply_neg[i], reply_context_candidates[i], reply_sim[i])
                reply_partcontext_list.append(reply_neg_partcontext)
                label_list.append(0)
            if hard_construction:
                # pos_reply:neg_reply = 1:2
                pos_reply_contrast_1 = self._selectblock_con(reply_pos, reply_pos_context, self.config.pos1_block_size,
                                                             self.config.pos1_block_num, noisy)
                pos_reply_contrast_2 = self._selectblock_con(reply_pos, reply_pos_context, self.config.pos2_block_size,
                                                             self.config.pos2_block_num, noisy)
                pos_reply_contrast_3 = self._selectblock_con(reply_pos, reply_pos_context, self.config.pos3_block_size,
                                                             self.config.pos3_block_num, noisy)
                reply_partcontext_list.append(pos_reply_contrast_1)
                reply_partcontext_list.append(pos_reply_contrast_2)
                reply_partcontext_list.append(pos_reply_contrast_3)
                label_list.append(1)
                label_list.append(1)
                label_list.append(1)
                neg_reply_contrast_1 = self._selectblock_con(reply_neg[0], reply_context_candidates[0],
                                                             self.config.pos1_block_size,
                                                             self.config.pos1_block_num,
                                                             noisy)
                neg_reply_contrast_2 = self._selectblock_con(reply_neg[1], reply_context_candidates[1],
                                                             self.config.pos2_block_size,
                                                             self.config.pos2_block_num,
                                                             noisy)
                neg_reply_contrast_3 = self._selectblock_con(reply_neg[2], reply_context_candidates[2],
                                                             self.config.pos3_block_size,
                                                             self.config.pos3_block_num,
                                                             noisy)
                neg_reply_contrast_4 = self._selectblock_con(reply_neg[3], reply_context_candidates[3],
                                                             self.config.block_num,
                                                             self.config.block_size,
                                                             noisy)
                reply_partcontext_list.append(neg_reply_contrast_1)
                reply_partcontext_list.append(neg_reply_contrast_2)
                reply_partcontext_list.append(neg_reply_contrast_3)
                reply_partcontext_list.append(neg_reply_contrast_4)
                label_list.append(0)
                label_list.append(0)
                label_list.append(0)
                label_list.append(0)
            reply_list.append(reply_partcontext_list)
            quo_list.append(quo_partcontext)
        return quo_list, reply_list, label_list

    def _selectblock(self, query, context, noisy):
        if noisy == 'RandomWordAug':
            aug_random = naw.RandomWordAug()
            query = aug_random.augment(query)[0]
            context = aug_random.augment(context)[0]
        if noisy == 'BackTranslationAug':
            aug_backtranlation = naw.BackTranslationAug()
            query = aug_backtranlation.augment(query)[0]
            context = aug_backtranlation.augment(context)[0]
        if noisy == 'KeyboardAug':
            aug_key = nac.KeyboardAug()
            query = aug_key.augment(query)[0]
            context = aug_key.augment(context)[0]
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

    def _selectblock_con(self, query, context, pos_block_size, pos_block_num, noisy):
        if noisy == 'RandomWordAug':
            aug_random = naw.RandomWordAug()
            query = aug_random.augment(query)[0]
            context = aug_random.augment(context)[0]
        if noisy == 'BackTranslationAug':
            aug_backtranlation = naw.BackTranslationAug()
            query = aug_backtranlation.augment(query)[0]
            context = aug_backtranlation.augment(context)[0]
        if noisy == 'KeyboardAug':
            aug_key = nac.KeyboardAug()
            query = aug_key.augment(query)[0]
            context = aug_key.augment(context)[0]
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

    def load_train_and_dev_files(self, train_file, dev_file, noisy=False):
        print('Loading train records for train...')
        train_set = self.load_file(train_file, noisy)
        print(len(train_set), 'training records loaded.')
        print('Loading dev records...')
        dev_set = self.load_file(dev_file, noisy)
        print(len(dev_set), 'dev records loaded.')
        return train_set, dev_set

    def load_valid_files(self, valid_file, noisy=False):
        print('Loading valid records...')
        valid_set = self.load_file(valid_file, noisy)
        print(len(valid_set), 'valid records loaded.')
        return valid_set

    def load_file(self, file_path='train.txt', noisy=False) -> TensorDataset:
        quo_list, reply_list, label_list = self._load_file(file_path, noisy)
        dataset = self._convert_sentence_pair_to_bert(
            quo_list, reply_list, label_list)
        return dataset

    def _augnoisy(self, text, noisy):
        if noisy != False:
            augmented_text = text
        if noisy == 'RandomWordAug':
            aug_random = naw.RandomWordAug()
            augmented_text = aug_random.augment(text)[0]
        if noisy == 'BackTranslationAug':
            aug_backtranlation = naw.BackTranslationAug()
            augmented_text = aug_backtranlation.augment(text)[0]
            # print('augmented_text_BackTranslationAug', augmented_text)
        if noisy == 'KeyboardAug':
            aug_key = nac.KeyboardAug()
            augmented_text = aug_key.augment(text)[0]
        return augmented_text

    def _load_file(self, filename, noisy):
        data_frame = pd.read_csv(filename, sep='#', header=None)[:500]
        quo_list, reply_list, label_list = [], [], []
        for index, row in data_frame.iterrows():
            candidates = list(row[4::2])
            quo_tokens = row[1]
            quo_tokens = self._augnoisy(quo_tokens, noisy)
            reply_label = row[2]
            reply_label = self._augnoisy(reply_label, noisy)
            reply_list_every_row = []
            reply_list_every_row.append(reply_label)
            label_list.append(1)
            for i in range(len(candidates)):
                reply_tokens_no_label = candidates[i]
                reply_tokens_no_label = self._augnoisy(reply_tokens_no_label, noisy)
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
