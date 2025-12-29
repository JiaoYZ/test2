from gensim.models import KeyedVectors
import numpy as np
import os

# 将题目的.emb转化为矩阵 embedding是题目的
# 输入文件地址，嵌入转换输出保存地址，嵌入生成类型标识，嵌入行数，嵌入维度
def emb_transfer(read_path, save_path, tp, emb_size, emb_dim):
    if tp == 'rand':
        vectors_np = np.random.normal(size=(emb_size, emb_dim)).astype(np.float32)
    else:
        # 处理有些题目不在训练集中的情况
        # 使用gensim工具读入emb文件，非二进制格式
        wv_from_text = KeyedVectors.load_word2vec_format(read_path, binary=False)
        # 统计实际出现在Word2Vec词汇表中的节点数量，有些节点可能没有被访问到或者随机游走到 最后 num_nodes和emb_size相比 若差距不大 说明效果较好
        num_nodes = len(wv_from_text.vocab)
        # 生成一个全0矩阵，用于存储嵌入向量  形状为(emb_size, emb_dim)
        vectors_np = np.zeros(shape=(emb_size, emb_dim), dtype=np.float32)
        # eval(vocab)将字符串vocab转换为整数，因为vocab是题目ID的字符串表示 ,将vocab行 填入wv_from_text.get_vector(vocab)那一行内容
        # 将wv_from_text.get_vector(vocab)转换为列表，因为wv_from_text.get_vector(vocab)是一个numpy数组
        for vocab in wv_from_text.vocab.keys():
            vectors_np[eval(vocab)] = list(wv_from_text.get_vector(vocab))
        print("total question number：%d, actual question number：%d" % (emb_size, num_nodes))
    # 保存到save_path   vectors_np是最终的npy文件里的元素
    np.save(save_path, vectors_np)


if __name__ == '__main__':
    for data_set in ["assist09_hkt"]:
        for MP in ['qtq']:
            for t in ['noWgt']:
                for numWalks in [10, 15]:
                    for walkLength in [80, 100]:
                        for dim in [128]:
                            for window_size in [3, 5]:
                                wv_path = "../%s/emb/%s_%s_%d_%d_%d_%d.emb" \
                                          % (data_set, MP, t, numWalks, walkLength, dim, window_size)
                                emb_transfer(wv_path, wv_path + '.npy', t, 18209, dim)

