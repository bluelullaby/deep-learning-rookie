from random import shuffle, randint, choice, sample
import numpy as np


def next_batch_sequence(data, batch_size, max_len=50):
    training_data = [item[1] for item in data.original_seq]
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1, data.item_num + 1))
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end - ptr, max_len), dtype=int)
        # 每个序列从1开始编号的位置信息
        pos = np.zeros((batch_end - ptr, max_len), dtype=int)
        # 与对应的序列的x的后一个元素
        y = np.zeros((batch_end - ptr, max_len), dtype=int)
        neg = np.zeros((batch_end - ptr, max_len), dtype=int)
        seq_len = []
        for n in range(0, batch_end - ptr):
            # 如果序列长度超过max_len, 取最后的max_len个元素
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end = len(training_data[ptr + n]) > max_len and max_len - 1 or len(training_data[ptr + n]) - 1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1, end + 1))
            y[n, :end] = training_data[ptr + n][start + 1:]
            negatives = sample(item_list, end)
            # 拿出和序列x中的item完全不同没有交集的同等数量的元素
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) > 0:
                negatives = sample(item_list, end)
            neg[n, :end] = negatives
        ptr = batch_end
        # 序列x, 序列元素的位置标号, 序列后一个元素y列表, 同等数量的负样本, 序列的长度
        yield seq, pos, y, neg, np.array(seq_len, int)


def next_batch_sequence_for_test(data, batch_size, max_len=50):
    sequences = [item[1] for item in data.original_seq]
    ptr = 0
    data_size = len(sequences)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end - ptr, max_len), dtype=int)
        pos = np.zeros((batch_end - ptr, max_len), dtype=int)
        seq_len = []
        for n in range(0, batch_end - ptr):
            start = len(sequences[ptr + n]) > max_len and -max_len or 0
            end = len(sequences[ptr + n]) > max_len and max_len or len(sequences[ptr + n])
            seq[n, :end] = sequences[ptr + n][start:]
            seq_len.append(end)
            pos[n, :end] = list(range(1, end + 1))
        ptr = batch_end
        yield seq, pos, np.array(seq_len, int)
