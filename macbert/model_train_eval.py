import time

import torch.optim
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
from data_preprocessing import get_dataloader
from classification_model import BertClassificationModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
config=Config()


def eval_model(model):
    # 加载验证集数据
    dev_dataloader=get_dataloader(config.dev_csv_path,shuffle=False)
    # 修改为评估模式
    model.eval()
    # 定义评估参数
    total_perd_result=[]
    total_true_result=[]
    # 模型预测
    for idx,(input_ids,attention_mask,token_type_ids,label) in enumerate(tqdm(dev_dataloader),start=1):
        input_ids=input_ids.to(device=config.device)
        attention_mask=attention_mask.to(device=config.device)
        token_type_ids=token_type_ids.to(device=config.device)
        label=label.to(device=config.device)
        with torch.no_grad():
            output=model(input_ids,attention_mask,token_type_ids)
        output=output.argmax(dim=-1)
        total_perd_result.extend(output.cpu().numpy())
        total_true_result.extend(label.cpu().numpy())

    f1=f1_score(total_true_result,total_perd_result,average='macro')
    precision=precision_score(total_true_result,total_perd_result,average='macro')
    recall=recall_score(total_true_result,total_perd_result,average='macro')
    return f1,precision,recall



def train_model():
    # 加载数据加载器
    train_dataloader=get_dataloader(config.train_csv_path)
    # 定义模型
    model=BertClassificationModel().to(device=config.device)
    # 定义优化器
    optimizer=optim.Adam(model.parameters(),lr=2e-5)
    # 定义损失函数
    loss=nn.CrossEntropyLoss()
    # 修改模型为训练模式
    model.train()

    # 定义训练参数
    epochs=1
    print_step=100
    best_f1=0.0
    for epoch in range(epochs):
        total_loss=0.0 # 总损失
        for idx,(input_ids,attention_mask,token_type_ids,label) in enumerate(tqdm(train_dataloader),start=1):
            input_ids=input_ids.to(device=config.device)
            attention_mask=attention_mask.to(device=config.device)
            token_type_ids=token_type_ids.to(device=config.device)
            label=label.to(device=config.device)
            # 开始训练
            output=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            # 计算损失
            my_loss=loss(output,label)
            total_loss+=my_loss.item()
            # 反向传播
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 输出训练结果
            if idx%print_step==0 or idx==len(train_dataloader):
                # 模型评估
                f1,precision,recall=eval_model(model)
                print(f"轮次：{epoch+1}，第{idx}步，损失：{total_loss/idx}，F1：{f1}，精确率：{precision}，召回率：{recall}")
                if best_f1<f1:
                    best_f1=f1
                    torch.save(model.state_dict(),config.classification_model_path)
                # 将模型恢复为训练模式
                model.train()

if __name__ == '__main__':
    train_model()



