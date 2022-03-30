from tqdm import tqdm
import os
import pandas as pd
from sklearn.model_selection import KFold

#读取训练文本数据
def agg_essays(train_flg, data_dir):
    folder = 'train' if train_flg else 'test'
    names, texts =[], []
    for f in tqdm(list(os.listdir(f'{data_dir}/{folder}'))):
        names.append(f.replace('.txt', ''))
        texts.append(open(f'{data_dir}/{folder}/' + f, 'r').read())
        df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts

#读取训练标签数据
def ner(df_texts, df_train):
    all_entities = []
    for _,  row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        total = len(row['text_split'])
        entities = ['O'] * total
        for _, row2 in df_train[df_train['id'] == row['id']].iterrows():
            discourse = row2['discourse_type']
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
            entities[list_ix[0]] = f'B-{discourse}'
            for k in list_ix[1:]: 
                entities[k] = f'I-{discourse}'
            entities_str = '#'.join(entities)
        all_entities.append(entities_str)

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts

#将数据和标签处理好放在一个dataframe
def preprocess(data_dir, df_train = None, remove_wrong_labels=True):
    if df_train is None:
        train_flg = False
    else:
        train_flg = True

    df_texts = agg_essays(train_flg, data_dir)

    if train_flg:
        df_texts = ner(df_texts, df_train)
        
    return df_texts

#对数据集进行切分
def split_fold(df_train,n_fold):
    ids = df_train['id'].unique()
    kf = KFold(n_splits=n_fold, shuffle = True, random_state=42)
    for i_fold, (_, valid_index) in enumerate(kf.split(ids)):
        df_train.loc[valid_index,'fold'] = i_fold
    return df_train

if __name__ == '__main__':
    data_dir = '/root/projects/feedback_prize/data'
    n_fold = 10
    df_train = pd.read_csv('/root/projects/feedback_prize/data/corrected_train.csv')
    train_texts = preprocess(data_dir,df_train)
    df_train = split_fold(train_texts, n_fold)
    df_train.to_csv(f'/root/projects/feedback_prize/data/process_data/all_train_texts_{n_fold}.csv')