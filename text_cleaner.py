# -*- coding: utf-8 -*-
# @Time    : 2021/7/13 12:57
# @Author  : Li Daji
# @File    : text_cleaner.py

from types import MethodType, FunctionType
import jieba

# 导入用于繁体/简体转换的包
from langconv import *


def clean_txt(raw):
    fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    return fil.sub(' ', raw)


def seg(sentence, sw, apply=None, cut_all=False):
    """
    对中文文本去特殊符号、去停用词、分词
    :param sentence: 原始中文文本
    :param sw:
    :param apply:
    :param cut_all:
    :return: 分词后中文文本
    """
    if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
        sentence = apply(sentence)
    return ' '.join([i for i in jieba.cut(sentence, cut_all=cut_all) if i.strip() and i not in sw])


def stop_words():
    with open('stopwords.txt', 'r', encoding='utf-8') as swf:
        return [line.strip() for line in swf]


def cht_to_chs(line):
    """
    中文繁体文本转简体
    :param line: 原始文本
    :return: 中文简体文本
    """
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


def replace_text(input_str, str_targ, str_rep):
    if isinstance(input_str, list):
        return [replace_text(s, str_targ, str_rep) for s in input_str]
    return input_str.replace(str_targ, str_rep)


# 对某个sentence进行处理：
if __name__ == '__main__':
    content = '海尔（Haier）新风机 室内外空气交换 恒氧新风机 XG-100QH/AA'
    res = seg(content.lower().replace('\n', ''), stop_words(), apply=clean_txt)
    print(res)
