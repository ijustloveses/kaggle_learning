# encoding: utf-8
"""
     这个脚本和 02 不同，要求每个 submission 中第二个字段是 test case 的打分
     也就是说，是一个概率值；那么这个脚本会根据这个概率值进行排序
     排序后，这个概率值就不用了，后面会根据排序的序号来为全部测试样本做 calibration
     Drawback : Averaging predictions often reduces overfit.
"""
from __future__ import division
from collections import defaultdict
from glob import glob
import sys

glob_files = sys.argv[1]
loc_outfile = sys.argv[2]


def kaggle_bag(glob_files, loc_outfile):
    """
        glob_files : pattern of submission files
        method & weights : now fixed to the given default value
    """
    with open(loc_outfile, "wb") as outfile:
        all_ranks = defaultdict(list)
        for i, f in enumerate(glob(glob_files)):
            file_ranks = []
            print 'parsing: ', f
            # 每个文件先收集结果
            for e, line in enumerate(open(f)):
                if i == 0 and e == 0:
                    outfile.write(line)   # header line
                elif e > 0:  # only handle the data lines
                    row = line.strip().split(',')
                    # 收集结果，列数，Id
                    file_ranks.append((float(row[1]), e, row[0]))
            # 然后按结果的概率值排序，这个概率值不再使用，根据排序号做调整
            for rank, item in enumerate(sorted(file_ranks)):
                # 看到 all_ranks 中每个测试样本保存在各个 submission 中的位次
                all_ranks[(item[1], item[2])].append(rank)
        # 收集完毕，开始做 calibration
        average_ranks = []
        for k in sorted(all_ranks):
            average_ranks.append((sum(all_ranks[k]) / len(all_ranks[k]), k))
        ranked_ranks = []
        for rank, k in enumerate(sorted(average_ranks)):
            # TODO: k[1][0], k[1][1] ?? why 2-dimension ??
            ranked_ranks.append((k[1][0], k[1][1], rank/(len(average_ranks) - 1)))
        for k in sorted(ranked_ranks):
            outfile.write("%s,%s\n" % (k[1], k[2]))
        print "Done !"

kaggle_bag(glob_files, loc_outfile)
