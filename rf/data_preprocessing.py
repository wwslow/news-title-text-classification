import json

import pandas as pd
from datasets import load_dataset
import os


def convert_arrow_to_csv(arrow_path, csv_path):
    """
    将arrow格式的数据集转换为csv格式

    Args:
        arrow_path: arrow文件路径
        csv_path: 输出的csv文件路径

    Returns:
        pd.DataFrame: 转换后的DataFrame
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 使用datasets库加载arrow文件
    dataset = load_dataset('arrow', data_files={'data': arrow_path})['data']

    # 转换为pandas DataFrame
    df = dataset.to_pandas()

    # 保存为csv
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存 {len(df)} 条数据到: {csv_path}")

    return df
def extract_classes_from_json(dataset_info_path, class_path):
    """
    从dataset_info.json中提取类别信息并生成class.txt文件

    Args:
        dataset_info_path: dataset_info.json文件路径
        class_path: 输出的class.txt文件路径

    Returns:
        dict: 类别名称到ID的映射
        dict: ID到类别名称的映射
    """
    # 读取dataset_info.json
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    # 提取类别信息
    label_feature = info['features']['label']
    class_names = label_feature['names']


    # 确保输出目录存在
    os.makedirs(os.path.dirname(class_path), exist_ok=True)

    # 保存为class.txt，格式：类别名称\t类别ID
    with open(class_path, 'w', encoding='utf-8') as f:
        for idx, name in enumerate(class_names):
            f.write(f"{name}\t{idx}\n")

    print(f"检测到 {len(class_names)} 个类别:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print(f"类别信息已保存到: {class_path}")



if __name__ == '__main__':
    from config import Config

    config = Config()

    # # 转换训练集
    # convert_arrow_to_csv(config.train_arrow_path, config.train_csv_path)
    #
    # # 转换验证集
    # convert_arrow_to_csv(config.dev_arrow_path, config.dev_csv_path)
    #
    # # 转换测试集
    # convert_arrow_to_csv(config.test_arrow_path, config.test_csv_path)
    extract_classes_from_json(config.dataset_info_path, config.class_path)
