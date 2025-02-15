from itertools import count
from numpy import char, unicode_
import numpy
import torch
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def load_books(path: str) -> list[str]:
    books = []
    # 定义正则表达式，匹配中文字符、标点符号和换行符
    pattern = re.compile(r"[^\u4e00-\u9fa5，。！？、；：“”（）《》\n]")
    # 定义英文标点符号和对应的中文标点符号
    punctuation_map = str.maketrans(
        {
            ",": "，",
            ".": "。",
            "?": "？",
            "!": "！",
            ";": "；",
            ":": "：",
            '"': "“",
            "(": "（",
            ")": "）",
            "<": "《",
            ">": "》",
        }
    )

    # 遍历指定路径下的所有文件
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 将英文标点符号转换为中文标点符号
                    content = content.translate(punctuation_map)
                    # 使用正则表达式替换掉不需要的字符
                    cleaned_content = re.sub(pattern, "", content)
                    books.append(cleaned_content)

    return books


def load_en_books(path: str) -> list[list[str]]:
    books = []
    # 仍旧先去除无关字符
    pattern = re.compile(r"[^a-z\s.,!?;:'\"()\n-]")

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                    cleaned_content = re.sub(pattern, "", content)

                    # 使用正则表达式分割，保留标点和纯空白（含换行、空格等）
                    tokens = re.split(r'([.,!?;:\'\"()\-]+|\s+)', cleaned_content)
                    
                    # 仅去除空字符串（即''），保留空白符号、标点等
                    # 如果想彻底去除空白，可改为 tk.strip() 过滤
                    tokens = [tk for tk in tokens if tk != '']

                    books.append(tokens)

    return books


def tokenize(books: list[str]):
    return [list(book) for book in books]

def en_tokenize(text: str) -> list[str]:
    # 定义正则表达式模式，匹配单词、标点符号和空白字符
    pattern = re.compile(r'([.,!?;:\'\"()\-]+|\s+|\w+)')
    
    # 使用正则表达式分割文本
    tokens = re.findall(pattern, text)
    
    # 过滤掉空字符串
    tokens = [tk for tk in tokens if tk != '']
    
    return tokens

class Vocab:
    def __init__(self, tokenized_books, min_freq=0):
        self.tokens = [c for book in tokenized_books for c in book]
        self.token_counts = Counter(self.tokens)
        self.sorted_token_counts = self.token_counts.most_common()
           
        self.unk = 0
        self.uniq_tokens = ["<unk>"]
        for token in self.sorted_token_counts:
            if token[1] >= min_freq:
                self.uniq_tokens.append(token[0])
        self.vocab_size = len(self.uniq_tokens)
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in self.uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple, numpy.ndarray)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple, numpy.ndarray)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def plot_token_frequencies(self):
        # 将词频排序，并只取前100个
        sorted_token_counts = self.token_counts.most_common(60)
        tokens, counts = zip(*sorted_token_counts)
        # 设置字体
        plt.rcParams["font.family"] = "WenQuanYi Micro Hei"  # 选择一个支持中文的字体

        # 可视化词频
        plt.figure(figsize=(15, 7))  # 调整图形大小
        plt.bar(tokens, counts)
        plt.xlabel("Tokens")
        plt.ylabel("Frequencies")
        plt.title("Top 100 Token Frequencies")
        plt.yscale("log")  # 设置 y 轴为对数尺度
        plt.xticks(rotation=90, fontsize=8)  # 旋转 x 轴标签并设置字体大小
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.show()


if __name__ == "__main__":
    books = load_en_books("/root/projs/python/mytorch/enbooks/1")
    books = tokenize(books=books)
    vocab = Vocab(tokenized_books=books)
    print(vocab['that'])
    print(vocab.__len__())
    
    
