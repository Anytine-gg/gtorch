from itertools import count
from numpy import char, unicode_
from sympy import N
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


def tokenize(books: list[str]):
    return [list(book) for book in books]


class Vocab:
    def __init__(self, books, min_freq=10):
        self.tokens = [c for book in books for c in book]
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
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
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
    books = load_books("/root/projs/py/demo/books")
    vocab = Vocab(books=books)
    print(vocab[['我','是','大','傻','逼']])
