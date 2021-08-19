# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 11:46
# @Author  : Wei Bangning
# @File    : fasttext_main.py

import fasttext
import pandas as pd
from sklearn.metrics import classification_report
import os
import time

report_index = 0


def train_model(train_file, dim=100, epoch=100, lr=0.5, loss='softmax', wordNgrams=2, save_dir=''):
    """
    训练fasttext模型并保存在 save_dir 文件夹， 详细参数参阅
    https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters
    :param train_file: 训练数据文件
    :param dim: 词向量大小， 默认100
    :param epoch: 默认100
    :param lr: 学习率， 默认0.5
    :param loss: 损失函数，默认softmax, 多分类问题推荐 ova
    :param wordNgrams: 默认2
    :param save_dir: 模型保存文件夹，默认不保存
    :return: 文本分类器模型
    """
    classifier = fasttext.train_supervised(train_file, label='__label__', dim=dim, epoch=epoch,
                                           lr=lr, wordNgrams=wordNgrams, loss=loss)
    if len(save_dir):
        model_name = f'model_dim{str(dim)}_epoch{str(epoch)}_lr{str(lr)}_loss{str(loss)}' \
                     f'_ngram{str(wordNgrams)}_{str(report_index)}.model'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        classifier.save_model(os.path.join(save_dir, model_name))
    return classifier


def give_classification_report(classifier, valid_csv, col_label='label', col_text='text', report_file=''):
    """
    使用 classification_report 验证 fasttext 模型分类效果，需在_FastText 类中添加dict_args()属性
    :param classifier: fasttext文本分类模型
    :param valid_csv: 验证数据集，需要csv格式
    :param col_label: 标签列名，默认 'label'
    :param col_text: 文本列名, 默认'text'
    :param report_file: 保存report文件名，默认不保存
    :return: classification report
    """
    df_valid = pd.read_csv(valid_csv, engine='python', encoding='utf-8')
    df_valid["predicted"] = df_valid[col_text].apply(lambda x: classifier.predict(str(x))[0][0])
    report = classification_report(df_valid[col_label].tolist(), df_valid["predicted"].tolist())
    model_args = classifier.dict_args()  # 需在_FastText 类中添加dict_args()属性
    if len(report_file):
        if not os.path.exists('report'):
            os.mkdir('report')
        report_file = os.path.join('report', report_file)
        with open(report_file, 'w') as f:
            print(time.asctime(time.localtime(time.time())), file=f)
            print(model_args, file=f)
            print(valid_csv, file=f)
            print('\n', file=f)
            print(report, file=f)
    return report


if __name__ == '__main__':
    print(' ')
