# 预训练模型+自定义网络层-->分类的目的
import torch
import torch.nn as nn
from transformers import  AutoModel, AutoConfig
from data_preprocessing import get_dataloader
from config import Config
config=Config()
# bert_config=BertConfig.from_pretrained(config.bert_path)
macbert_config=AutoConfig.from_pretrained(config.macbert_path)


class BertClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=AutoModel.from_pretrained(config.macbert_path).to(device=config.device)
        # # 预训练模型不更新参数
        for param in self.bert.parameters():
            param.requires_grad=False
        # 解冻最后 N 层
        N = 2
        total_layers = macbert_config.num_hidden_layers
        for i in range(total_layers - N, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        self.linear=nn.Linear(macbert_config.hidden_size, len(config.id2class))

    def forward(self,input_ids,attention_mask,token_type_ids):
        output=self.bert(input_ids,attention_mask,token_type_ids)
        # print(output)
        output=self.linear(output["pooler_output"])
        return output
if __name__ == '__main__':
    dataloader=get_dataloader(config.train_csv_path)
    model=BertClassificationModel()
    for input_ids,attention_mask,token_type_ids,label in dataloader:
        output=model(input_ids,attention_mask,token_type_ids)
        print(f"输出的向量维度：{output.shape}")
        break