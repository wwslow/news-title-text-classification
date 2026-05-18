

# 新闻标题文本分类

基于深度学习和机器学习的新闻标题文本分类系统，支持 BERT、BERT-BiLSTM、随机森林等多种模型。

## 项目简介

本项目实现了一个新闻标题文本分类系统，可将新闻标题自动分类到不同的类别。支持多种模型架构，包括：

- **BERT 分类模型** - 基于预训练 BERT 的文本分类
- **BiLSTM 模型** - 双向 LSTM 分类器
- **模型蒸馏** - 将 BERT 模型知识蒸馏到轻量级 BiLSTM 模型
- **随机森林** - 传统机器学习方法

## 功能特性

- 支持训练和评估多种分类模型
- 提供 Flask RESTful API 服务
- Web 端预测界面
- 模型蒸馏功能，将大模型知识迁移到小模型

## 数据说明

数据文件位于 `data/` 目录：

- `train.csv` - 训练数据集
- `dev.csv` - 验证数据集
- `test.csv` - 测试数据集
- `class.txt` - 类别标签文件

## 环境依赖

```
torch
transformers
flask
scikit-learn
pandas
numpy
```

## 快速开始

### 1. 训练模型

```bash
# 训练 BERT 分类模型
cd macbert
python model_train_eval.py

# 训练随机森林模型
cd rf
python rf_train_eval.py
```

### 2. 启动 API 服务

```bash
cd macbert
python service_api.py
```

### 3. 使用 Web 预测界面

启动服务后，访问 `html/index.html` 即可使用可视化预测界面。

## 项目结构

```
.
├── data/                    # 数据集
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   └── class.txt
├── data_show/              # 数据可视化
│   └── show_data.py
├── html/                   # Web 前端
│   └── index.html
├── macbert/                # BERT 系列模型
│   ├── classification_model.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model_train_eval.py
│   ├── model_distill.py
│   ├── service_api.py
│   ├── student_model.py
│   └── BiLSTM_predict_service.py
└── rf/                     # 随机森林模型
    ├── config.py
    ├── data_preprocessing.py
    └── rf_train_eval.py
```

## API 使用

发送 POST 请求到 `/predict` 端点：

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "你的新闻标题"}'
```

## 许可证

MIT License