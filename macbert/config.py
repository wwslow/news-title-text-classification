import torch


class Config:
    def __init__(self):

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device='cpu'
        # 原始json格式路径
        self.train_arrow_path = '../data/thuc_news-train.arrow'
        self.test_arrow_path= '../data/thuc_news-test.arrow'
        self.dev_arrow_path= '../data/thuc_news-validation.arrow'
        self.dataset_info_path = '../data/dataset_info.json'

        # csv格式路径
        self.train_csv_path = '../data/train.csv'
        self.test_csv_path= '../data/test.csv'
        self.dev_csv_path= '../data/dev.csv'

        # 类别文件路径
        self.class_path= '../data/class.txt'

        # id2class
        self.id2class={idx:line.strip() for idx,line in enumerate(open(self.class_path, encoding='utf-8').readlines())}

        # 停用词路径
        self.stop_words_path = '../data/stopwords.txt'


        # Bert预训练模型路径
        self.bert_path=r"D:\codeSave\python\nlp\transfer_learning\pre_model\bert-base-chinese"
        self.macbert_path=r'D:\huggingface\hub\models--hfl--chinese-macbert-base\snapshots\a986e004d2a7f2a1c2f5a3edef4e20604a974ed1'

        # 模型保存路径
        self.classification_model_path='model/bert_model.pth'

        # 学生模型保存路径
        self.student_model_path='model/student_model.pth'

        # 量化后模型保存路径
        self.quantization_model_path='model/quantization_model.pth'

        # 模型长度
        self.max_len=30

if __name__ == '__main__':
    config=Config()
    print(config.id2class)
