#Pytorch-SMFF实现 #

## 数据集准备
* [MIND新闻推荐数据集](https://msnews.github.io/)
    - small dev https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
    - small train https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
* Glove 预训练词向量, 
    - https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.42B.300d.zip

## 数据处理
1.创建文件夹
```angular2html
    cd data                
    mkdir mind             # The path to the MIND dataset
    mkdir glove            # The path of the glove word vector
    mkdir processedData    # The path of the Preprocessed files
```
2.预处理数据集
```angular2html
    python ProcessMindByGlove.py
```

## 训练模型
1.训练模型
```angular2html
    nohup python -u train.py >train.txt
```
2.查看验证结果
```angular2html
    cat train.txt
```
