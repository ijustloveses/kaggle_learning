# encoding: utf-8
"""
     operates on a list of Kaggle submissions and create a new one
     就是每个test case，从各个 submission file 中收集标签，最后取其中最多的
     这个脚本要求每个 submission 文件中第二个字段是每个 test case 的标签
     Drawback : Averaging predictions often reduces overfit.
"""
from collections import defaultdict, Counter
from glob import glob
import sys

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]


def kaggle_bag(glob_files, loc_outfile, method='average', weights='uniform'):
    """
        glob_files : pattern of submission files
        method & weights : now fixed to the given default value
    """
    if method == 'average':
        scores = defaultdict(list)
    with open(loc_outfile, "wb") as outfile:
        for i, f in enumerate(glob(glob_files)):
            print 'parsing: ', f
            for e, line in enumerate(open(f)):
                if i == 0 and e == 0:
                    outfile.write(line)   # header line
                if e > 0:  # only handle the data lines
                    row = line.strip().split(',')
                    scores[(e, row[0])].append(row[1])
        for j, k in sorted(scores):
            # 从结果标签列表中，取出最多的一个标签，显然无权重概念
            outfile.write("%s,%s\n" % (k, Counter(scores[(j, k)]).most_common(1)[0][0]))
        print "Done !"

kaggle_bag(glob_files, loc_outfile)
