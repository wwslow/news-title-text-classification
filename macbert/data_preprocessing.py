

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
import json

from transformers import AutoTokenizer



from transformers import BertTokenizer
config=Config()

my_tokenizer = AutoTokenizer.from_pretrained(config.macbert_path)

# 对数据集进行读取
def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 对text进行截断,超过30个字则截断
    df['text'] = df['text'].apply(lambda x: x[:30] if len(x)>30 else x)
    # print(list(zip(df['text'], df['label'])))
    return list(zip(df['text'], df['label']))


# 自定义dataset对象
class MyDataset(Dataset):
    def __init__(self,data_list):
        self.data=data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item=max(0,min(item,len(self.data)-1))
        return self.data[item]

def collate_fn(data):
    # print(data)
    # print(list(zip(*data)))
    x,label=zip(*data)
    x_tensor=my_tokenizer.batch_encode_plus(
        x,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # print(x_tensor)
    input_ids=x_tensor['input_ids']
    attention_mask=x_tensor['attention_mask']
    token_type_ids=x_tensor['token_type_ids']
    label_tensor=torch.tensor(label,dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, label_tensor



# 构建dataloader对象
def get_dataloader(csv_path, batch_size=32, shuffle=True):
    dataset = MyDataset(load_data(csv_path))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

# 使用示例
if __name__ == '__main__':
    # dataset=MyDataset(load_data(config.train_csv_path))
    # print(dataset[0])
    dataloader=get_dataloader(config.train_csv_path)
    for data in dataloader:
        print(data)
        break
    ...