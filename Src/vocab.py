from collections import Counter
from tqdm import tqdm
import jieba


def generate_vocab_file(train_file, test_file, val_file, vocab_file_path):
    """
    从三个文本文件中生成词汇表
    :param train_file: 训练数据文件路径
    :param test_file: 测试数据文件路径
    :param val_file: 验证数据文件路径
    :param vocab_file_path: 词汇表保存路径
    :param vocab_size: 词汇表大小
    """
    all_words = []

    # 读取训练数据文件
    process(all_words, train_file)
    process(all_words, test_file)
    process(all_words, val_file)

    counter = Counter(all_words)

    # 动态计算词汇表大小
    vocab_size = 200000

    # 获取词频最高的前 vocab_size 个词语作为词汇表
    count_pairs = counter.most_common(vocab_size)
    words, _ = zip(*count_pairs)
    words = ['<PAD>'] + list(words)

    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))

    print(f"已保存包含 {vocab_size} 个词的词汇表文件至 {vocab_file_path}")


def process(all_words, file):
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                category, content = line.strip().split('\t')
                content = content.replace('\n', '').replace('\t', '').replace('\u3000', '')
                words = jieba.lcut(content)
                all_words.extend(words)


generate_vocab_file("D:\\MyCode\\Graduation project\\Data\\cnews.train.txt",
                    "D:\\MyCode\\Graduation project\\Data\\cnews.test.txt",
                    "D:\\MyCode\\Graduation project\\Data\\cnews.val.txt",
                    '.\\data\\cnews\\vocab.txt')
