#Pytorch-CMM实现 #

## 数据集准备
* MIND新闻推荐数据集，下载链接：https://msnews.github.io/
* Glove 预训练词向量。

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
    cd CMM
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
