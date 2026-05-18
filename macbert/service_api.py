# 对外的接口
from flask import Flask, request, jsonify
from flask_cors import CORS
from macbert.BiLSTM_predict_service import predict


app=Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_api():
    """
    文本分类预测接口
    请求格式: {"text": "新闻标题内容"}
    返回格式: {"text": 新闻标题内容, "pred_class": 预测类型}
    """
    # 获取json参数
    data=request.get_json()
    # 调用预测服务
    result=predict(data)
    # 返回结果，确保结果为json格式
    return jsonify(result)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)