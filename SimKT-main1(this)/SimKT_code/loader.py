import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
import numpy as np

import os

DEVICE = None

# 加载数据
def load_data(args):
    # 初始化四个字典，文件地址、数据列表、数据集、数据加载
    filePath_dict, dataList_dict, dataSet_dict, dataLoader_dict = dict(), dict(), dict(), dict()
    # 训练集和测试集是否打乱
    shuffle = {'train': True, 'test': False}
    # 使用参数中的设备
    global DEVICE
    DEVICE = torch.device(args.device)
    # 遍历训练集和测试集
    for train_or_test in ['train', 'test']:
        # 保存文件地址
        filePath_dict[train_or_test] = os.path.join(args.data_path, args.data_set, "train_test",
                                                    train_or_test + '_%s.txt' % args.input)
        # 传入参数文件地址 最小序列长度 最大序列长度 下面的file_to_list函数
        dataList_dict[train_or_test] = file_to_list(filePath_dict[train_or_test], args.min_seq_len, args.max_seq_len)
        # 下面的KTDataset类 传入参数 序列长度列表 问题id列表 答案列表
        dataSet_dict[train_or_test] = KTDataset(dataList_dict[train_or_test][0], dataList_dict[train_or_test][1],
                                                dataList_dict[train_or_test][2])
        # 输入的数据 batch_size 默认为8 collate 将不同的序列 整合为一个batch 具体函数在下面
        dataLoader_dict[train_or_test] = DataLoader(dataSet_dict[train_or_test], batch_size=args.batch_size,
                                                    collate_fn=collate_fn, shuffle=shuffle[train_or_test])
    # 获取 映射的问题数量 
    with open(os.path.join(args.data_path, args.data_set, "encode", "question_id_dict.txt")) as f:
        dataLoader_dict["num_ques"] = len(eval(f.read()).keys())
    q_mat = np.load(os.path.join(args.data_path, args.data_set, "graph", "ques_skill_mat.npy"))
    dataLoader_dict["num_skill"] = q_mat.shape[1]
    # 将q矩阵 转换为 张量 并移动到 gpu上
    dataLoader_dict["q_mat"] = torch.from_numpy(q_mat).to(DEVICE)

    print("number of question=%d" % dataLoader_dict["num_ques"])
    print("number of skill=%d" % dataLoader_dict["num_skill"])
    print('load data done!')

    # dataLoader_dict 包含 训练集和测试集的 数据加载器 问题数量 技能数量 问题技能矩阵
    return dataLoader_dict

# 传入参数 文件地址 最小序列长度 最大序列长度 是否截断 默认为false
# 将文件内容转换为 序列长度列表 问题id列表 答案列表
def file_to_list(filename, min_seq_len=3, max_seq_len=200, truncate=False):
    # 拆分长度函数，输入为序列长度
    def split_func(_seq_len):
        # 拆分长度列表
        _split_list = []
        # 如果序列长度大于0 开始拆分
        while _seq_len > 0:
            # 如果长度大于最大长度 默认是200
            if _seq_len >= max_seq_len:
                # 先将最大长度加入列表
                _split_list.append(max_seq_len)
                # 计算剩余长度
                _seq_len -= max_seq_len
            # 如果剩余长度大于等于最小长度 但不大于最大长度
            elif _seq_len >= min_seq_len:
                # 将该长度加入列表
                _split_list.append(_seq_len)
                _seq_len -= _seq_len
            else:
                # 这是剩余的长度，小于最小长度，直接丢掉
                _seq_len -= min_seq_len
        # 返回拆分长度列表的长度 和 长度列表
        return len(_split_list), _split_list
    # 初始化序列长度列表 题目id列表 答案列表
    seq_lens, ques_ids, answers = [], [], []
    # 记录拆分段数
    k_split = -1
    # 读取文件内容
    with open(filename) as file:
        lines = file.readlines()
    i = 0
    # 开始处理文件内容
    while i < len(lines):
        # 去除无效字符串
        line = lines[i].rstrip()
        # i mod 3 == 0 是长度行
        if i % 3 == 0:
            # 转换为整数 这是序列长度
            seq_len = int(line)
            # 如果序列长度小于最小长度 直接跳过
            if seq_len < min_seq_len:
                i += 3
                continue
            # 如果长度合适开始拆分
            else:
                # 拆分段数 拆分长度列表
                k_split, split_list = split_func(seq_len)
                # 如果截断为true 只取第一段
                if truncate:
                    k_split = 1
                    seq_lens.append(split_list[0])
                # 不直接截断
                else:
                    # 将拆分长度列表加入序列长度列表  这里相当于append操作
                    seq_lens += split_list
        # 非长度行
        else:
            # 将行里的每个元素 用 ‘ , ’ 隔开 转换为整数列表
            line = line.split(',')
            # 将 行中每个元素 转换为整数 并加入列表
            array = [int(eval(e)) for e in line]
            # 如果是 题目 id 行或者skill id 行
            if i % 3 == 1:
                # 按照拆分段数来进行拆分
                for j in range(k_split):
                    # 问题序列列表，嵌套列表 将array里面的元素 加入到 问题序列列表 中  多余的也不会越界或这溢出
                    # 运算表达式[0-199]  [200-399] max_seq_len 就是 200
                    ques_ids.append(array[max_seq_len * j: max_seq_len * (j + 1)])
            # 如果是作答结果行 同上
            else:
                for j in range(k_split):
                    answers.append(array[max_seq_len * j: max_seq_len * (j + 1)])
        i += 1
    # for integrity, check the lengths
    # 检查 序列长度列表 问题id列表 答案列表 长度是否相同  问题id列表里面的元素也是列表 每个元素列表是作答问题的id
    assert len(seq_lens) == len(ques_ids) == len(answers)
    # 返回长度列表 问题id列表 答案列表
    return seq_lens, ques_ids, answers


class KTDataset(Dataset):
    def __init__(self, seq_lens, ques_ids, answers):
        self.seq_lens = seq_lens
        self.ques_ids = ques_ids
        self.answers = answers

    def __len__(self):
        return len(self.seq_lens)
    # 获取元素的方法
    def __getitem__(self, item):
        seq_len = self.seq_lens[item]
        ques_id = self.ques_ids[item]
        answer = self.answers[item]
        # 样本长度 序列长度-1
        sample_len = torch.tensor([seq_len - 1], dtype=torch.long)
        # 取前n-1个问题id 作为输入 最后一个作为预测的问题id
        sample_exercise = torch.tensor(ques_id[:-1], dtype=torch.long)
        # 取前n-1个作答结果 作为输入 最后一个作为预测的作答结果
        sample_answer = torch.tensor(answer[:-1], dtype=torch.long)
        # 从第2个开始 后n-1个问题id 作为预测的问题id
        sample_next_exercise = torch.tensor(ques_id[1:], dtype=torch.long)
        # 后n-1个作答结果 作为预测的作答结果
        sample_next_answer = torch.tensor(answer[1:], dtype=torch.float)
        # 2 3 用于模型推理  4 5 用于模型标签验证
        return sample_len, sample_exercise, sample_answer, sample_next_exercise, sample_next_answer


def collate_fn(batch):
    # Sort the batch in the descending order
    # 对批次内按照 样本长度 进行排序 
    batch = sorted(batch, key=lambda x: x[0], reverse=True)
    # 将排序后的 每个元素的长度 拼接为一个张量
    seq_lens = torch.cat([x[0] for x in batch])
    # 将排序后的 问题序列 填充成相同长度 没有的填充成0
    questions = pad_sequence([x[1] for x in batch], batch_first=False)
    # 答案序列 填充成相同长度 没有的默认填充成0
    answers = pad_sequence([x[2] for x in batch], batch_first=False)
    # 将排序后的 下一个问题id 填充成相同长度
    next_questions = pad_sequence([x[3] for x in batch], batch_first=False)
    # 提取每个样本的下一个答案序列，打包成PackedSequence格式 就是标签
    next_answers = pack_sequence([x[4] for x in batch], enforce_sorted=True)
    # 将 序列长度 问题序列 答案序列 下一个问题id 下一个答案 移动到 gpu上
    return [i.to(DEVICE) for i in [seq_lens, questions, answers, next_questions, next_answers]]

