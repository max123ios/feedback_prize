import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import gc
# from pytorch_lightning_spells.optimizers import Lookahead
from Lookahead import Lookahead
from utils_data.dict_data import *
from utils_data.metrics import score_feedback_comp

#这里只用到了处理数据的id
def process_pred_ids(ids, all_preds, all_preds_prob):
    final_preds = []
    for i in range(len(ids)):
        idx = ids[i]
        pred = all_preds[i]
        pred_prob = all_preds_prob[i]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1 #这里为什么要＋1，加过了是有问题的
            else: cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls !='':
                avg_score = np.mean(pred_prob[j:end]) #计算这个平均概率
                if end - j > MIN_THRESH[cls] and avg_score > PROB_THRESH[cls]:
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'predictionstring']
    return df_pred

#这里只用到了处理数据的id
def process_pred(df, all_preds, all_preds_prob):
    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_preds[i]
        pred_prob = all_preds_prob[i]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1 #这里为什么要＋1，加过了是有问题的
            else: cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls !='':
                avg_score = np.mean(pred_prob[j:end]) #计算这个平均概率
                if end - j > MIN_THRESH[cls] and avg_score > PROB_THRESH[cls]:
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'predictionstring']
    return df_pred

def preds_class_prob(all_logits, word_ids):
    # print("predict target class and its probabilty")
    final_predictions = []
    final_predictions_score = []
    for i in range(all_logits.shape[0]):
        word_id = word_ids[i].cpu().numpy()
        predictions =[]
        predictions_prob = []
        pred_class_id = np.argmax(all_logits[i].cpu().numpy(), axis=1)
        pred_score = np.max(all_logits[i].cpu().numpy(), axis=1)
        pred_class_labels = [IDS_TO_LABELS[j] for j in pred_class_id]
        prev_word_idx = -1
        for idx, word_idx in enumerate(word_id):
                if word_idx == -1:
                    pass
                elif word_idx != prev_word_idx:
                    predictions.append(pred_class_labels[idx])
                    predictions_prob.append(pred_score[idx])
                    prev_word_idx = word_idx
        final_predictions.append(predictions)
        final_predictions_score.append(predictions_prob)
    return final_predictions, final_predictions_score

def active_logits(raw_logits, word_ids, num_labels):
    word_ids = word_ids.view(-1)
    active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], num_labels) #expand是复制
    active_mask = active_mask != NON_LABEL
    active_logits = raw_logits.view(-1, num_labels)
    #把所有不是非补充值挑出来，构成一个一维Tensor
    active_logits = torch.masked_select(active_logits, active_mask) # return 1dTensor
    #由于非补充值，一排都非补充值，因此，这样形状转换没有问题
    active_logits = active_logits.view(-1, num_labels)
    #展平把padding的词去除
    return active_logits
    
def active_labels(labels):
    active_mask = labels.view(-1) != IGNORE_INDEX
    active_labels = torch.masked_select(labels.view(-1), active_mask)
        #展平把padding的词给去除
    return active_labels
    
def active_preds_prob(active_logits):
    #挑出概率最大的值以及概率最大值的位置
    active_preds = torch.argmax(active_logits, axis = 1)
    active_preds_prob, _ = torch.max(active_logits, axis = 1)
    return active_preds, active_preds_prob

class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(FeedbackModel, self).__init__()
        model_config = AutoConfig.from_pretrained(model_name)
        # model_config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=model_config)
        self.model_config = model_config
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        #这里添加了lstm
        # self.lstm = nn.LSTM(model_config.hidden_size, model_config.hidden_size // 2,
        #                     num_layers=1, bidirectional=True, batch_first = True )
        self.head = nn.Linear(model_config.hidden_size, num_labels)
    
    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        #这里用来添加lstm
        # x = self.lstm(x[0])
        logits = self.head(x[0])
        # logits1 = self.head(self.dropout1(x[0]))
        # logits2 = self.head(self.dropout2(x[0]))
        # logits3 = self.head(self.dropout3(x[0]))
        # logits4 = self.head(self.dropout4(x[0]))
        # logits5 = self.head(self.dropout5(x[0]))
        # logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits
    

class FeedbackTrainModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, learning_rate, df_train, df_val):
        super().__init__()
        self.lr = learning_rate
        self.num_labels = num_labels
        self.model = FeedbackModel(model_name, self.num_labels)
        self.df_train = df_train
        self.df_val = df_val
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        # self.automatic_optimization = False
    
    def share_step(self, batch):
        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']
        raw_logits = self.model(input_ids, att_mask)
        word_ids = batch['word_ids']
        targets = batch['labels']
        
        #把一个batch的预测值展平，并去掉不需要计分的补充值
        logits = active_logits(raw_logits, word_ids, self.num_labels)
        sf_logits = torch.softmax(logits, dim=-1)
        #找到概率最大的位置，找到概率最大的值
        preds, preds_prob = active_preds_prob(sf_logits)
        #把一个batch的标签值展平，并去掉
        labels = active_labels(targets)
        train_accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        loss = self.criterion(logits, labels)
        del input_ids, att_mask, logits, preds, preds_prob, targets
        torch.cuda.empty_cache()
        gc.collect()
        return {'loss': loss, 'accuracy':train_accuracy, 'raw_logits':raw_logits}

    def training_step(self, batch, batch_idx):
        # opt_a = self.optimizers(use_pl_optimizer=True)
        # opt_a.zero_grad()
        result = self.share_step(batch)
        # loss = result['loss']
        # loss.backward()
        # opt_a.step()
        self.log('train_loss', result['loss'])
        self.log('train_acc', result['accuracy'])
        return result['loss']
    
    #计算每一个batch的f1score
    def cal_f1score(self, ids, raw_logits, word_ids, df_csv):
        label_csv = df_csv.query('id==@ids').reset_index(drop=True)
        final_predictions, final_predictions_score = preds_class_prob(raw_logits, word_ids)
        pred_df = process_pred_ids(ids, final_predictions, final_predictions_score)
        f1 = score_feedback_comp(pred_df, label_csv)
        return f1


    def validation_step(self, batch, batch_idx):
        result = self.share_step(batch)
        f1 = self.cal_f1score(batch['id_text'], result['raw_logits'], batch['word_ids'], self.df_val)
        self.log('val_loss', result['loss'])
        self.log('val_acc', result['accuracy'])
        self.log('val_f1_score', f1)
        return {
            'val_loss': result['loss'], 
            'val_acc':result['accuracy'], 
            'f1_score':f1
        # 'ids' : batch['id_text'],
        # 'raw_logits' : result['raw_logits'],
        # 'word_ids' : batch['word_ids']
                }
    
    # def validation_epoch_end(self, validation_step_outputs):
    #     raw_logits = []
    #     word_ids = []
    #     ids = []
    #     #这里可以提取已经处理的文本的id
    #     for i in validation_step_outputs:
    #         raw_logits.append(i['raw_logits'])
    #         word_ids.append(i['word_ids'])
    #         ids = ids + i['ids']
    #     raw_logits = torch.cat(raw_logits, dim = 0) #模型的直接输出结果
    #     word_ids = torch.cat(word_ids, dim = 0)
    #     f1 = self.cal_f1score(ids, raw_logits, word_ids, self.df_val)
    #     self.log('val_f1_score', f1)
    #     print(f'val_f1_score%f1:{f1}')
   
    def configure_optimizers(self):
        # return Lookahead(torch.optim.SGD(self.parameters(), momentum=0.9, lr=0.1),alpha=0.5, k=6, pullback_momentum="pullback")
            
        # return Lookahead(torch.optim.RAdam(self.parameters(), lr=self.lr), alpha=0.5, k=6)
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
        # return [torch.optim.RAdam(self.parameters(), lr=self.lr)],[Lookahead(torch.optim.RAdam(self.parameters(), lr=self.lr),alpha=0.5, k=6)]

def create_model(cfg, df_train, df_val):
    model = FeedbackTrainModel(cfg.model_name, cfg.num_labels, cfg.learning_rate, df_train, df_val)
    return model 