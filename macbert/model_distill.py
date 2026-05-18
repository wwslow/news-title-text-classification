import torch.optim
from sklearn.metrics import f1_score, precision_score, recall_score
from config import Config
from data_preprocessing import get_dataloader
from classification_model import BertClassificationModel
from macbert.model_train_eval import eval_model
from student_model import StudentBiLSTMModel
import torch.nn as nn
from  torch.nn.functional import kl_div
import torch.optim as optim
from tqdm import tqdm
config=Config()



def distill_model():
    # 加载数据加载器
    train_dataloader=get_dataloader(config.train_csv_path)
    # 定义学生模型
    student_model=StudentBiLSTMModel().to(device=config.device)
    # 加载老师模型
    teacher_model=BertClassificationModel().to(device=config.device)
    teacher_model.load_state_dict(torch.load(config.classification_model_path))
    # 定义优化器
    optimizer=optim.Adam(student_model.parameters())
    # 定义损失函数
    loss=nn.CrossEntropyLoss()
    # 修改学生模型为训练模式
    student_model.train()
    # 修改老师模型为评估模式
    teacher_model.eval()
    # 定义训练参数
    epochs=5
    print_step=100
    best_f1=0.0

    # 软标签参数
    alpha=0.7 # 软标签的权重
    T=3.0 # 温度参数
    # 开始训练
    for epoch in range(epochs):
        total_loss=0.0
        for idx,(input_ids,attention_mask,token_type_ids,label) in enumerate(tqdm(train_dataloader),start=1):
            input_ids=input_ids.to(device=config.device)
            attention_mask=attention_mask.to(device=config.device)
            token_type_ids=token_type_ids.to(device=config.device)
            label=label.to(device=config.device)
            # 老师前向传播
            with torch.no_grad():
                teacher_output=teacher_model(input_ids,attention_mask,token_type_ids)

            # 学生前向传播
            student_output=student_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            input=torch.log_softmax(student_output/T,dim=-1)
            target=torch.softmax(teacher_output/T,dim=-1)
            #KL散度就是软标签的损失值
            '''
                input: 学生模型的输出结果
                target：教师模型的数据结果
                reduction: 指定返回的张量形状，"batchmean"表示对批次维度求平均
                log_target: 指定目标张量是否已经经过对数处理
            '''
            kl_loss=kl_div(
                input=input,
                target=target,
                reduction='batchmean',
                log_target=True
            )

            # 硬标签损失
            hard_loss=loss(student_output,label)
            # 总损失计算
            distill=(1-alpha) * hard_loss+ alpha* T**2 * kl_loss
            total_loss+=distill.item()
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            distill.backward()
            # 参数更新
            optimizer.step()
            # 第一轮不进行评估
            if (idx%print_step==0 or idx==len(train_dataloader)) and epoch>1:
                # 模型评估
                f1,precision,recall=eval_model(student_model)
                print(f"轮次：{epoch+1}，第{idx}步，损失：{total_loss/idx}，F1：{f1}，精确率：{precision}，召回率：{recall}")
                if best_f1<f1:
                    best_f1=f1
                    torch.save(student_model.state_dict(),config.student_model_path)
                # 将模型恢复为训练模式
                student_model.train()

if __name__ == '__main__':
    distill_model()