# 用来多多个模型进行对比，包括macbert模型，蒸馏模型

import torch
from model_train_eval import eval_model
from config import Config
from classification_model import BertClassificationModel
from student_model import StudentBiLSTMModel

config=Config()
if __name__ == '__main__':
    model=BertClassificationModel()
    student_model=StudentBiLSTMModel()
    model.load_state_dict(torch.load(config.classification_model_path))
    student_model.load_state_dict(torch.load(config.student_model_path))
    model.eval()
    student_model.eval()
    f1,precision,recall=eval_model(model)
    f1_student,precision_student,recall_student=eval_model(student_model)
    print(f"bert模型：F1：{f1}，精确率：{precision}，召回率：{recall}")
    print(f"蒸馏模型：F1：{f1_student}，精确率：{precision_student}，召回率：{recall_student}")