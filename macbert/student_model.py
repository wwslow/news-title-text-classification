import torch
import torch.nn as nn
from config import Config
from transformers import AutoConfig,AutoTokenizer
from data_preprocessing import get_dataloader
config=Config()
macbert_config=AutoConfig.from_pretrained(config.macbert_path)
macbert_tokenizer=AutoTokenizer.from_pretrained(config.macbert_path)
# 自定义学生模型
class StudentBiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size=macbert_config.vocab_size
        self.embed_size=128
        self.hidden_size=256
        self.num_layers=3

        # 词嵌入层
        self.embed=nn.Embedding(self.vocab_size, self.embed_size)

        # BiLSTM层
        self.biLSTM=nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,bidirectional=True)

        # 3.随机失活层
        self.dropout=nn.Dropout(p=0.3)
        # 全连接层
        self.linear=nn.Linear(self.hidden_size*2, len(config.id2class))

    def forward(self,input_ids,attention_mask,token_type_ids):
        # 词嵌入编码
        input_embed=self.embed(input_ids)

        # 生成掩码编码
        # 去掉无意义的词汇信息，包括[CLS]、[SEP]、[PAD]
        cls_index=macbert_tokenizer.cls_token_id
        sep_index=macbert_tokenizer.sep_token_id
        pad_index=macbert_tokenizer.pad_token_id
        # 过滤掉[SEP]和[CLS]
        embed_mask=(input_ids!=cls_index)&(input_ids!=sep_index)
        # 过滤掉[PAD]
        embed_mask=embed_mask&attention_mask
        # 进行升维 [batch_size,seq_len]->[batch_size,seq_len,1]
        embed_mask=torch.unsqueeze(embed_mask,dim=-1)
        # 对词向量进行掩码
        input_embed=input_embed*embed_mask

        # 进行BiLSTM层 output->[batch_size,seq_len,embed_size]
        output, (_, _) = self.biLSTM(input_embed)

        # 对output进行平均池化
        output=torch.sum(output,dim=1) #[batch_size,embed_size]

        # 获取token的个数
        token_size=torch.sum(embed_mask,dim=1)
        output=output/token_size
        # 进行全连接层
        return self.linear(self.dropout(output))
