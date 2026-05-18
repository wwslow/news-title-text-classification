# 使用随机森林进行base-line
import pickle

from sklearn.metrics import f1_score, precision_score, recall_score

from config import Config
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF类

from sklearn.ensemble import RandomForestClassifier
import jieba

def chinese_tokenizer(text):
    return ' '.join(jieba.cut(text))
config=Config()
# 加载数据
data_train=pd.read_csv(config.train_csv_path)

data_dev=pd.read_csv(config.dev_csv_path)
# print(len(data_train))
x_train=data_train['text']
y_train=data_train['label']
# print(len(data_dev))
x_test=data_dev['text']
y_test=data_dev['label']
# 数据预处理
# 截断过长的句子，max_length=30
x_train=x_train.apply(lambda x: x[:30] if len(x)>30 else x)
x_test=x_test.apply(lambda x: x[:30] if len(x)>30 else x)
# 使用前先分词
x_train = x_train.apply(chinese_tokenizer)
x_test = x_test.apply(chinese_tokenizer)
# # 划分训练集和测试集
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=21)
# 特征工程
# 创建TF-IDF向量化器
tfidf=TfidfVectorizer()
# 对数据进行向量化处理
x_train=tfidf.fit_transform(x_train)
x_test=tfidf.transform(x_test)
# 模型训练
model=RandomForestClassifier(n_estimators=100,n_jobs=10)
model.fit(x_train,y_train)
# 模型预测
y_pred=model.predict(x_test)
print("准确率:",model.score(x_test,y_test))
print("精确率:",precision_score(y_test,y_pred,average="macro"))
print("召回率:",recall_score(y_test,y_pred,average="macro"))
print("F1-score:",f1_score(y_test,y_pred,average="macro"))
# 保存模型
with open(config.rf_model_path,mode="wb") as f:
    pickle.dump(model,f)

