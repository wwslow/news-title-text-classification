# 对模型进行DQ量化

import torch
from model_train_eval import eval_model
from config import Config

from student_model import StudentBiLSTMModel

config=Config()
if __name__ == '__main__':
    # 加载模型
    student_model=StudentBiLSTMModel()
    student_model.load_state_dict(torch.load(config.student_model_path))
    # student_model.eval()
    # f1,precision,recall=eval_model(student_model)
    # print(f"量化前模型：F1：{f1}，精确率：{precision}，召回率：{recall}")

    # 进行DQ量化
    # 创建量化配置
    qconfig_spec={torch.nn.Linear,torch.nn.LSTM} # 指定需要量化的层
    quantize_model=torch.quantization.quantize_dynamic(student_model,qconfig_spec=qconfig_spec, dtype=torch.qint8)
    # 进行量化后评估
    quantize_model.eval()
    f1,precision,recall=eval_model(quantize_model)
    print(f"量化后模型：F1：{f1}，精确率：{precision}，召回率：{recall}")
    torch.save(quantize_model, config.quantization_model_path)

