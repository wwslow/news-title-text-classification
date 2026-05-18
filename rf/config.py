import torch


class Config:
    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # 停用词路径
        self.stop_words_path = '../data/stopwords.txt'
        # 随机森林模型保存路径
        self.rf_model_path = 'model/rf_model.pkl'


if __name__ == '__main__':
    config=Config()
