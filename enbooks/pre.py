import os
import re
import chardet  # 如果未安装，请使用 pip install chardet


# 目录下是一些英文txt,读取这些文件预处理，其中把大写变为小写，
# 将多个连续空格变为一个空格，将多个换行变为一个换行，保留标点符号，非英语字符，非acsii字符删除，
# 用utf8保存。原txt可能不是utf8，打开记得注意一下。最后在这个文件夹下生成output文件夹，
# 输出处理过的txt


def detect_encoding(filepath):
    with open(filepath, "rb") as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result["encoding"]


def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 将多个连续空格变为一个空格
    text = re.sub(r"[ ]+", " ", text)
    # 将多个换行变为一个换行
    text = re.sub(r"\n+", "\n", text)
    # 删除非 ASCII 字符 (保留标点、数字、字母、空格等ascii字符)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


def process_files(input_folder):
    output_folder = os.path.join(input_folder, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # 避免处理自己生成的 output 文件夹
        filepath = os.path.join(input_folder, filename)
        if os.path.isdir(filepath) and filename == "output":
            continue
        if filename.lower().endswith(".txt"):
            try:
                encoding = detect_encoding(filepath)
                with open(filepath, "r", encoding=encoding, errors="ignore") as f:
                    content = f.read()
                processed = preprocess_text(content)
                out_path = os.path.join(output_folder, filename)
                with open(out_path, "w", encoding="utf8") as f:
                    f.write(processed)
                print(f"Processed file: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    # 假设文件均在当前工作目录，如果需要可以修改为其他目录
    current_folder = "/Users/guoziye/Library/CloudStorage/OneDrive-个人/code/multiplatform/Python/mytorch/enbooks/1"
    process_files(current_folder)
