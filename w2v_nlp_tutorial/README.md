官方资料
---------
https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words


MLWave 的 Perceptron 脚本
-------------------------
http://mlwave.com/online-learning-perceptron/
https://github.com/MLWave/online-learning-perceptron/
    We do generate 2-grams (such as “not good”) from the features, much like the script from Abhishek does. 
    We simply hash these 2-grams and add them to the sample vectors.
对应 perceptron.py 文件

其中还提到了 Abhishek 基于 word2vec + linear regression 实现的一个 baseline 脚本
对应 linear_by_Abhishek.py 文件


bag of word 模型
-----------------
https://github.com/zygmuntz/classifying-text
http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/
见 classifying-text 目录，其中修改了一些脚本获取 data 的路径 

这个目录下还有根据论文 
A Fixed-Size Encoding Method for Variable-Length Sequences with its Application to Neural Network Language Models 
(http://arxiv.org/abs/1505.01504)
实现的 fofe.py，这是一种 slightly better than a vanilla count vectorizer, but worse than TF-IDF 的编码
不过，知道就好，我们不深入研究了

故此，只需要关心以下几个脚本
bow_predict.py - train and predict, save a submission file
bow_validate.py - create train/test split, train, get validation score
bow_validate_tfidf.py - an improved validation script, with TF-IDF and n-grams

