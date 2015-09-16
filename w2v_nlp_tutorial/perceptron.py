# encoding: utf-8

"""
    https://www.kaggle.com/c/word2vec-nlp-tutorial/data
    脚本使用了 hash trick，把词做了 hash，另外加入 2-gram 一起作为特征
    然后运行一个简单的 n-passes perceptron，没有做 cv, 也没有做任何 anti-overfitting
    不使用提供的 unlabeledTrainData.tsv，因为不会使用这个来做无监督学习，或者学习 word2vec
"""

import re
import random
from math import log
from datetime import datetime
from operator import itemgetter


def clean(str):
    return " ".join(re.findall(r'\w+', str, flags=re.UNICODE | re.LOCALE)).lower()


def get_data_tsv(loc_dataset, opts):
    for e, line in enumerate(open(loc_dataset, "rb")):
        if e > 0:  # skip header line
            r = line.strip().split("\t")
            id = r[0]
            if opts["clean"]:
                try:
                    # training set 第三项是文字，第二项是 标签结果 (0 或者 1)
                    r[2] = clean(r[2])
                except:
                    # test set 第二项是文字
                    r[1] = clean(r[1])
            if len(r) == 3:
                # training set，文字 ==> [(hash1, 1), (hash2, 1), ..]
                # 这里要注意，hash1， hash2，这些是 feature； 后面的 1 表示这个文字中含有这个 feature
                features = [(hash(f) % opts["D"], 1) for f in r[2].split()]
                label = int(r[1])
            else:  # test set
                features = [(hash(f) % opts["D"], 1) for f in r[1].split()]
                label = 1
            # append 2-grams  elements
            for i in xrange(len(features) - 1):
                features.append((hash(str(features[i][0]) + str(features[i + 1][0])) % opts["D"], 1))
            yield label, id, features


# features 为所有的一元词，二元词，features 的最大值是 opts["D"]
# 那么， weight 很简单，就是一个 opts["D"] 维的向量
# 给了一个句子的 features，features 包含句子含有的全部 feature
# 于是含有的 feature 都是 1，那么不含的就是 0，于是也组成 opts["D"] 维的向量
def dot_product(features, weights):
    dotp = 0
    for f in features:
        dotp += weights[f[0]] * f[1]   # 其实 f[1] 就是 1
    return dotp


def train(loc_dataset, opts):
    start = datetime.now()
    print ("Pass\t\tErrors\t\tAverage\t\tNr. Samples\tSince Start")
    random.seed(3003)
    weights = [random.random()] * opts["D"]
    # 训练不止一遍
    for pass_nr in xrange(opts["n_passes"]):
        error_counter = 0
        for e, (label, id, features) in enumerate(get_data_tsv(loc_dataset, opts)):
            dp = dot_product(features, weights) > 0.5
            error = label - dp  # 0 means correct, 1 or -1 means mis-classified
            if error != 0:
                error_counter += 1
                # 更新权重
                for index, value in features:
                    weights[index] += opts["learning_rate"] * error * log(1. + value)
        # 到此，e 代表了文件中一共多少条数据 - 1
        print("%s\t\t%s\t\t%s\t\t%s\t%s" % (pass_nr + 1, error_counter,
              round(1 - error_counter / float(e + 1), 5), e + 1, datetime.now() - start))
        # earlier stop
        if error_counter == 0 or error_counter < opts["errors_satisfied"]:
            print "%s errors found during training, halting ..." % error_counter
            break
    return weights


def test(loc_dataset, weights, opts):
    preds = []
    for e, (label, id, features) in enumerate(get_data_tsv(loc_dataset, opts)):
        dotp = dot_product(features, weights)
        dp = dotp > 0.5
        if dp:
            preds.append([id, 1, dotp])
        else:
            preds.append([id, 0, dotp])
    # normalizing dotp
    max_dotp = max(preds, key=itemgetter(2))[2]
    min_dotp = min(preds, key=itemgetter(2))[2]
    for p in preds:
        p.append((p[2] - min_dotp) / float(max_dotp - min_dotp))
    print "Done !"
    return preds

if __name__ == '__main__':
    opts = {}
    opts["D"] = 2 ** 25
    opts["learning_rate"] = 0.1
    opts["n_passes"] = 80
    opts["errors_satisfied"] = 0
    opts["clean"] = True

    weights = train("labeledTrainData.tsv", opts)
    preds = test("testData.tsv", weights, opts)

    with open("popcorn_perceptron.csv", "wb") as outfile:
        outfile.write('"ID","Sentiment"\n')
        for p in sorted(preds):
            outfile.write("%s,%s\n" % (p[0], p[1]))
