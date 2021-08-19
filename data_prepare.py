# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 11:46
# @Author  : Wei Bangning
# @File    : data_prepare.py

import pandas as pd
import numpy as np
import random
from text_cleaner import *
from tqdm import tqdm


def load_df(file_path, encoding='utf-8', drop_dup=True, drop_na=True):
    """
    从csv文件读取dataframe
    :param file_path: csv文件路径
    :param encoding: 编码，默认 UTF-8
    :param drop_dup: 去掉重复行
    :param drop_na: 去掉空行
    :return: dataframe
    """
    df = pd.read_csv(file_path, encoding=encoding, engine='python')
    if drop_dup:
        df = df.drop_duplicates()
    if drop_na:
        df = df.dropna()
    return df


def write_txt(file_name, df_data, delimiter=' ', fmt="%s", encoding='utf-8'):
    """
    把dataframe写入txt，用于DF转fasttext训练集
    :param delimiter:
    :param file_name: 写入的txt文件路径
    :param df_data: <label> <text>型的DataFrame
    :param fmt: 格式，默认为字符串
    :param encoding: 编码，默认为 UTF-8
    :return:
    """

    np.savetxt(file_name, df_data.values, delimiter=delimiter, fmt=fmt, encoding=encoding)


def dataframe_split(df_text, train_ratio):
    """
    将dataframe按比例分割
    :param df_text: 原始dataframe
    :param train_ratio: 训练集占比
    :return: 训练集和验证集
    """
    train_set_size = int(len(df_text) * train_ratio)
    valid_set_size = int(len(df_text) * (1 - train_ratio))
    df_train_data = df_text[:train_set_size]
    df_valid_data = df_text[train_set_size:(train_set_size + valid_set_size)]
    return df_train_data, df_valid_data


def count_diff_in_col(df_text, col_name):
    """
    统计某一列不同种类的个数
    :param df_text: dataframe
    :param col_name: 需要统计的列名
    :return: 一个字典
    """
    col_set = set(df_text[col_name].values)
    col_list = list(df_text[col_name].values)
    compute = dict()
    for item in col_set:
        compute.update({item: col_list.count(item)})
    return dict(sorted(compute.items()))


def drop_rows_where_col_has(dataframe, col_name, target):
    """
    删除 dataframe中， col_name列包含target的行
    :param dataframe:
    :param col_name:
    :param target:
    :return: 新的dataframe
    """
    return dataframe.drop(dataframe[dataframe[col_name] == target].index)


def df_data_augmentation(dataframe, col_label='label', col_text='text', num_sample=50, sample_length=18):
    """
    将每一类标签的样本扩充至指定数量
    :param dataframe:
    :param col_label:
    :param col_text:
    :param num_sample: 扩充后每个种类样本的数量，默认50
    :param sample_length: 样本文本的长度， 默认18
    :return: 返回扩充后的dataframe 和 记录不同标签样本的字典
    """
    dict_tmp = count_diff_in_col(dataframe, col_label)
    df_sample = dataframe.copy(deep=True)
    for key in list(dict_tmp.keys()):
        if dict_tmp[key] < num_sample:
            df_tmp = df_sample[(df_sample[col_label] == key)]
            list_text = []
            for text in df_tmp[col_text].values.tolist():
                list_text.extend(text.split())
            while dict_tmp[key] < num_sample:
                str_tmp = ' '.join(random.sample(list_text, sample_length))
                df_sample = df_sample.append({col_label: key, col_text: str_tmp}, ignore_index=True)
                dict_tmp.update({key: dict_tmp[key] + 1})
    return df_sample, dict_tmp


def repalce_df_text(dataframe, col_name, str_targ, str_rep):
    """
    将dataframe中某一列的字符串中 str_tage 替换为 str_rep
    :param dataframe:
    :param col_name:
    :param str_targ:
    :param str_rep:
    :return:
    """
    li0 = dataframe[col_name].values.tolist()
    li1 = replace_text(li0, str_targ, str_rep)
    if len(li1) == len(li0):
        dataframe[col_name] = li1
        return dataframe
    else:
        print('Lenghth of dataframe has been changed !')
        return -1


def df_cut_ch(dataframe, col_name, save_path=''):
    """
    对 dataframe的col_name列中的中文文本分词， 默认cut_all
    :param dataframe:
    :param col_name:
    :param save_path:保存路径，默认不保存
    :return:
    """
    df_cut = dataframe.copy(deep=True)
    text_cut = []
    for text in tqdm(dataframe[col_name].values.tolist()):
        text_cut.append(seg(text.lower().replace('\n', ''), stop_words(), apply=clean_txt))
    del df_cut[col_name]
    df_cut[col_name] = text_cut
    if len(save_path):
        df_cut.to_csv(save_path, encoding='utf-8', index=False)
    return df_cut
