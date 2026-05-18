import torch
from transformers import AutoTokenizer

from config import Config
from student_model import StudentBiLSTMModel

config = Config()
# 直接加载量化模型对象
dq_model = torch.load(config.quantization_model_path, map_location=config.device)

# 设置分词器
tokenizer=AutoTokenizer.from_pretrained(config.macbert_path)
dq_model.eval()
def predict(data):
    '''
    用于对外提供预测服务
    :param data: data是dict,结构：{"title":新闻标题}
    :return: 返回结构：{"title":新闻标题，"pred_class":预测类型}
    '''
    # 健壮性检测
    if not isinstance(data,dict):
        return {"error":"参数必须是dict"}
    if not data.__contains__("title"):
        data["error"]='参数中，缺乏title'
        return data
    title=data["title"]
    x_tensor=tokenizer.batch_encode_plus(
        [title],
        max_length=config.max_len,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    input_ids=x_tensor['input_ids']
    attention_mask=x_tensor['attention_mask']
    token_type_ids=x_tensor['token_type_ids']
    model_output=dq_model(input_ids,attention_mask,token_type_ids)
    # print(model_output.argmax(dim=-1)) # tensor([5])
    class_idx=model_output.argmax(dim=-1).item()
    return {"title":title,"pred_class":config.id2class[class_idx]}

if __name__ == '__main__':
    print(predict({"title": "中国和日本开始对峙"}))
