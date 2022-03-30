from cProfile import label
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append('/root/projects/feedback_prize')
from utils_data.dict_data import *
from utils_data import  dataprocess
from transformers import AutoTokenizer, LongformerConfig, LongformerModel, LongformerTokenizerFast
import pytorch_lightning as pl
import pandas as pd

class FeedBackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, has_labels, label_subtokens):
        super().__init__()
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels #是否是训练或者验证的数据集，有标签处理
        self.label_subtokens = label_subtokens #是否记录拆分的单词
    
    def __getitem__(self, index):
        text = self.data.text[index]
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words = True,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_len
        )
        word_ids = encoding.word_ids()
        #处理标签
        if self.has_labels:
            word_labels = self.data.entities[index].split('#')
            prev_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(IGNORE_INDEX)
                elif word_idx != prev_word_idx:
                    label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
                else:
                    if self.label_subtokens:
                        label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
                    else:
                        label_ids.append(IGNORE_INDEX)
                prev_word_idx = word_idx
        encoding['labels'] = label_ids #单词的标签，正确的分类正常分类，补充的词用-100表示
        item = {k: torch.as_tensor(v) for k, v in encoding.items()}
        word_ids2 = [w if w is not None else NON_LABEL for w in word_ids]
        item['word_ids'] = torch.as_tensor(word_ids2) #单词/sub_token在句子中的位置，补充的词用-1表示
        item['id_text'] = self.data.id[index]
        item['text'] = text
        #想要什么额外信息，从这里返回
        return item

    def __len__(self):
        return self.len

def create_dataloader(**cfg): #需要传入的信息：model_name, train_dir, val_fold, batch_size, shuffle, num_workers, has_labels, label_subtokens
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], add_prefix_space = True)
    val_data = None
    val_label_csv = None
    if cfg['train_dir']:
        df_train_text = pd.read_csv(cfg['train_dir'])
        train_labe_csv = pd.read_csv(cfg['train_label'])
        if cfg['val_fold'] != -1:
            df_val_text = df_train_text[df_train_text['fold'] == cfg['val_fold']].reset_index(drop = True)
            df_train_text = df_train_text[df_train_text['fold'] != cfg['val_fold']].reset_index(drop = True)
            val_set = FeedBackPrizeDataset(dataframe = df_val_text, tokenizer = tokenizer, max_len = cfg['max_len'], has_labels = cfg['has_labels'], label_subtokens = cfg['label_subtokens'])
            val_data = DataLoader(val_set , batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=False)
            val_idlist = df_val_text['id'].unique().tolist()
            val_label_csv = train_labe_csv.query('id==@val_idlist').reset_index(drop=True)
            train_labe_csv = train_labe_csv.query('id!=@val_idlist').reset_index(drop=True)
    data_set = FeedBackPrizeDataset(dataframe = df_train_text, tokenizer = tokenizer, max_len = cfg['max_len'], has_labels = cfg['has_labels'], label_subtokens = cfg['label_subtokens'])
    train_data = DataLoader(data_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=False)
    return train_data, val_data, train_labe_csv, val_label_csv


if __name__ == '__main__':
    cfg = {
        'model_name' : 'allenai/longformer-base-4096',
        'train_dir' : '/root/projects/feedback_prize/data/process_data/all_train_texts_5.csv',
        'train_label' : '/root/projects/feedback_prize/data/train.csv',
        'batch_size' : 2,
        'num_workers' : 10,
        'has_labels' : True,
        'label_subtokens':True,
        'val_fold' : 0,
        'max_len' : 1000
    }
    print(cfg['model_name'])
    train_data, val_data, train_labe_csv, val_label_csv = create_dataloader(**cfg)
    # print(val_label_csv)
    for i in val_data:
        words_id = i['word_ids']
        print(words_id.shape) #(2,1000)
        words_id = i['word_ids'].view(-1)
        print(words_id.shape) #(2000,)
        # print(i['word_ids'].unsqueeze(1).expand(i['word_ids'].shape[0], 15)) #2000,15
        print(words_id.unsqueeze(1).expand(words_id.shape[0], 15).shape) #2000,15
        exit()


    # model_name = 'allenai/longformer-base-4096'
    # tokenizer = LongformerTokenizerFast.from_pretrained(model_name, add_prefix_space = True)
    # data_dir = '/root/projects/feedback_prize/data'
    # data = pd.read_csv('/root/projects/feedback_prize/data/process_data/all_train_texts_5.csv')
    # dataset = FeedBackPrizeDataset(data, tokenizer, 1600, True, True)
    # dataloder = DataLoader(dataset, batch_size=32)
    # for i in dataloder:
    #     print(i['input_ids'].shape)
    #     print(i['attention_mask'].shape)
    #     print(i['labels'].shape)
    #     print(i['word_ids'].shape)
    #     print(len(i['id_text']))
    #     break
