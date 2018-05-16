# -*- coding: utf-8 -*-
#wangyan
import jieba
import pandas as pd
from gensim.summarization import bm25

# 读取训练数据，并存到train_data
print("读取训练数据")
train_data = []
data1 = pd.read_csv("train_data.csv")
train_title = data1['title']
for each in train_title:
    train_data.append(each)

# 使用jieba对训练数据进行分词
train_data_list = []
for each in train_data:
    data_list = [word for word in jieba.cut(each)]
    train_data_list.append(data_list)



# 读取测试数据存到test_data，并将训练数据的id保存到test_id
print("读取测试数据")
test_data = []
data2 = pd.read_csv("test_data.csv", encoding="gbk")
test_title = data2["title"]
for each in test_title:
    test_data.append(each)
id2 = data2['id']
test_id = []
for each in id2:
    test_id.append(each)

# 测试文档进行分词
test_data_list = []
for each in test_data:
    test_list = [word for word in jieba.cut(each)]
    test_data_list.append(test_list)

# 制作语料库
print("制作语料库")
#dictionary = corpora.Dictionary(train_data_list)
#corpus = [dictionary.doc2bow(each) for each in train_data_list]  # (0,1)(1,1)

#map=0.114,在此基础上进行调参,使得结果到0.118
def bm25_test():
    bm25.PARAM_K1 = 2
    bm25.PARAM_B = 0.82
    bm25.EPSILON = 0.2
    #使用BM25训练模型
    bm25Model = bm25.BM25(train_data_list)
    # 计算平均逆文档频率
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    results = []
    for test in test_data_list:
        #计算每个测试数据与训练数据的相似性得分
        scores = bm25Model.get_scores(test, average_idf)     #[0,0,0.3289392]
        #相似性得分排序，取前21
        sim_sorted = sorted(enumerate(scores), key=lambda item: -item[1])[:21]
        #保留前21个最相似数据的序号
        indexs = [str(item[0] + 1) for item in sim_sorted]
        #全部加入列表results
        results.append(" ".join(indexs))
    #将每个测试数据最相似的21条写入results.txt,并与测试数据下标test_id对应
    with open("results.txt", "w") as f:
        count = 0
        for item in results:
            item = item.strip().split()
            for i in range(0, 21):
                f.write(str(test_id[count]) + "\t" + str(item[i]) + "\n")
            count += 1
    #按格式将数据写入submisson_new44，并删除Top21里与自己相同的那条数据，数据保存好啦
    with open("results.txt", "r") as f, open("submisson_new44.txt", "w") as file:
        file.write("source_id" + "\t" + "target_id" + "\n")
        datas = f.readlines()
        for data in datas:
            data = data.strip().split("\t")
            if data[0] != data[1]:
                file.write(data[0] + "\t" + data[1] + "\n")
if __name__ == "__main__":
    bm25_test()
