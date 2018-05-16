import snownlp

# list = ['xixi','shid','yuhjhn']
# docp = []
# for l in list:
#     doc=[]
#     doc.append(l)
#     docp.append(doc)
# print(docp)
import jieba
from gensim import corpora,similarities,models
from gensim.summarization import bm25


raw_documents = [
    '1无偿居间介绍买卖毒品的行为应如何定性',
    '2吸毒男动态持有大量毒品的行为该如何认定',
    '3如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
    '4为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
    '5将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
    '6为获报酬帮人购买毒品的行为该如何认定',
    '7毒贩出狱后再次够买毒品途中被抓的行为认定',
    '8虚夸毒品功效劝人吸食毒品的行为该如何认定',
    '9妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
    '10一方未签字办理的结婚登记是否有效',
    '11夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
    '12结婚前对方父母出资购买的住房写我们二人的名字有效吗',
    '13身份证被别人冒用无法登记结婚怎么办？',
    '14同居后又与他人登记结婚是否构成重婚罪',
    '15未办登记只举办结婚仪式可起诉离婚吗',
    '16同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
]

#训练文本使用jieba进行分词
print("训练文本使用jieba进行分词")
all_raw_list = []
for raw in raw_documents:
    raw_list = [word for word in jieba.cut(raw)]
    all_raw_list.append(raw_list)
print(all_raw_list)

#测试文本使用jieba进行分词
test_raw_list = []
print("测试文本使用jieba进行分词")
test_data = ['你好，我想问一下我想离婚他不想离，孩子他说不要，是六个月就自动生效离婚',
             '家人因涉嫌运输毒品被抓，她只是去朋友家探望朋友的，结果就被抓了，还在朋友家收出毒品，可家人的身上和行李中都没有。现在已经拘留10多天了，请问会被判刑吗'
             ]
for test in test_data:
    test_list = [word for word in jieba.cut(test)]
    test_raw_list.append(test_list)

print(test_raw_list)


#制作语料库
print("制作语料库")
dictionary = corpora.Dictionary(all_raw_list)
print(len(dictionary.keys()))
#corpus = [[(0,1),(1,1)],[],[]...,[]]对每个短句的词进行编码
corpus = [dictionary.doc2bow(doc) for doc in all_raw_list]

def tfidf():
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    sim = similarities.SparseMatrixSimilarity(corpus_tfidf,num_features=len(dictionary.keys()))

    #对每条测试数据进行编码[(0,1),(1,1)]
    results = []
    for test in test_raw_list:
        test1 = dictionary.doc2bow(test)
        #[(21, 0.29542256657146615), (96, 0.3567282882016457), (126, 0.8862676997143984)]
        test_tfidf = tfidf_model[test1]
        print(test_tfidf)

        #测试数据与原文档的相似度[0，0，0.0808069，0.34657258]
        test_sim = sim[test_tfidf]
        print(test_sim)
        test_sim_sorted = sorted(enumerate(test_sim), key=lambda item: -item[1])[:3]
        print(test_sim_sorted)
        indexs = [str(item[0]+1) for item in test_sim_sorted]
        print(indexs)
        results.append(" ".join(indexs))
        print(results)
    with open("results_prac_t.txt", "w") as f:
        j=0
        for item in results:
            item = item.strip().split()
            for i in range(0, 3):
                f.write(str(j) + "\t" + str(item[i]) + "\n")
            j+=1
        # print(list(enumerate(test_sim)))
def bm252():


    # vec1_sorted = sorted(vec1, key=lambda (x,y): y, reverse=True)
    # print(len(vec1_sorted))
    # for term, freq in vec1_sorted[:5]:
    #     print(dictionary[term])

    bm25Model = bm25.BM25(all_raw_list)
    print(bm25Model)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    print(average_idf)
    # query_str = '离婚 毒品 药物'
    # query = []
    results=[]
    for test in test_raw_list:
        # test1 = dictionary.doc2bow(test)
    # for word in query_str.strip().split():
    #     query.append(word)

        scores = bm25Model.get_scores(test, average_idf)
        print(scores)
        # scores.sort(reverse=True)
        # print(scores)
        test_sim_sorted = sorted(enumerate(scores), key=lambda item: -item[1])[:3]
        print(test_sim_sorted)
        indexs = [str(item[0]+1) for item in test_sim_sorted]
        print(indexs)
        results.append(" ".join(indexs))
        print(results)
    with open("results.txt", "w") as f:
        j = 0
        for item in results:
            item = item.strip().split()
            for i in range(0, 3):
                f.write(str(j) + "\t" + str(item[i]) + "\n")
            j += 1
    with open("results.txt", "r") as f, open("submisson6.txt", "w") as wf:
        wf.write("source_id" + "\t" + "target_id" + "\n")
        datas = f.readlines()
        for data in datas:
            data = data.strip().split("\t")
            if data[0] != data[1]:
                wf.write(data[0] + "\t" + data[1] + "\n")
if __name__ == "__main__":
    tfidf()
    bm252()