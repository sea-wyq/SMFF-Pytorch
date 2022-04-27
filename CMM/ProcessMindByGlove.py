import math
import random
import re

import numpy as np
import pandas as pd
import torch

from Config import Config


def getWordDictByGlove(config):
    word_dict = [np.zeros(config.FEATURE_EMB_LENGTH, dtype=np.int32)]
    word_list = []
    with open(config.GLOVE_PATH, 'r', encoding='utf—8') as glove:
        for line in glove.readlines():
            line = list(line.split())
            word = line[0]
            word_list.append(word)
            word_emb = torch.tensor([float(x) for x in line[1:]])
            word_dict.append(word_emb)
    glove_dict_length = len(word_dict)
    word2id = {word: id + 1 for id, word in enumerate(word_list)}
    word_dict = np.stack(word_dict, axis=0)
    np.save(config.PROCESS_DATA_PATH + "glove_dict.npy", word_dict)
    np.save(config.PROCESS_DATA_PATH + "word2id.npy", word2id)
    print("Glove的字典个数为: ", glove_dict_length)


def getCategoryAndNewsDict(config):
    train_news = pd.read_csv(config.DATA_PATH + "MIND" + config.DATASET_TYPE + "_train/news.tsv", sep="\t",
                             names=config.NEWS_NAME)
    dev_news = pd.read_csv(config.DATA_PATH + "MIND" + config.DATASET_TYPE + "_dev/news.tsv", sep="\t",
                           names=config.NEWS_NAME)
    cate_list = []
    subcate_list = []
    news_dict = {}
    for new in train_news.iterrows():
        cate = new[1]["Category"]
        if cate not in cate_list:
            cate_list.append(cate)
        subcate = new[1]["SubCategory"]
        if subcate not in subcate_list:
            subcate_list.append(subcate)
        news_dict[new[1]["NewsID"]] = {"Category": new[1]["Category"], "SubCategory": new[1]["SubCategory"],
                                       "Title": new[1]["Title"], "Abstract": new[1]["Abstract"]}

    for new in dev_news.iterrows():
        cate = new[1]["Category"]
        if cate not in cate_list:
            cate_list.append(cate)
        subcate = new[1]["SubCategory"]
        if subcate not in subcate_list:
            subcate_list.append(subcate)
        news_dict[new[1]["NewsID"]] = {"Category": new[1]["Category"], "SubCategory": new[1]["SubCategory"],
                                       "Title": new[1]["Title"], "Abstract": new[1]["Abstract"]}
    cate2id = {w: i + 1 for i, w in enumerate(cate_list)}

    subcate2id = {w: i + 1 for i, w in enumerate(subcate_list)}

    np.save(config.PROCESS_DATA_PATH + "cate2id.npy", cate2id)
    np.save(config.PROCESS_DATA_PATH + "subcate2id.npy", subcate2id)
    np.save(config.PROCESS_DATA_PATH + "news_dict.npy", news_dict)
    print("Cate Dict个数为: ", len(cate2id))
    print("SubCate Dict个数为：", len(subcate2id))
    print("News Dict的个数为：", len(news_dict))
    return cate2id, len(cate2id) + 1, subcate2id, len(subcate2id) + 1, news_dict


def getSentenceEmbByGlove(feature, new_id, news_dict, glove_id, feature_max_length):
    feature_list = []
    sentence = news_dict.get(new_id).get(feature)
    if pd.isnull(sentence):
        return torch.zeros((feature_max_length,), dtype=torch.int32)
    else:
        word_list = re.sub("[()|\'\":.,!?\\-]", '', sentence.lower()).split(' ')
        sentence_length = len(word_list)
        for j in range(feature_max_length):
            if j < sentence_length:
                feature_list.append(glove_id.get(word_list[j], glove_id.get("unk")))
            else:
                feature_list.append(0)
        return torch.as_tensor(feature_list, dtype=torch.int32)


def getHisEmbByGlove(new_list, news_dict, glove_id, cate_dict, subcate_dict, feature_max_length):
    tit_emb_list = []
    abs_emb_list = []
    cate_id_list = []
    subcate_id_list = []
    for new_id in new_list:
        if new_id == "EMP":
            tit_emb_list.append(torch.zeros((feature_max_length,), dtype=torch.int32))
            abs_emb_list.append(torch.zeros((feature_max_length,), dtype=torch.int32))
            cate_id_list.append(0)
            subcate_id_list.append(0)
        else:
            tit_emb_list.append(getSentenceEmbByGlove("Title", new_id, news_dict, glove_id, feature_max_length))
            abs_emb_list.append(getSentenceEmbByGlove("Abstract", new_id, news_dict, glove_id, feature_max_length))
            cate_id_list.append(cate_dict.get(news_dict.get(new_id).get("Category")))
            subcate_id_list.append(subcate_dict.get(news_dict.get(new_id).get("SubCategory")))
    tit_emb_list = torch.stack(tit_emb_list)
    abs_emb_list = torch.stack(abs_emb_list)
    cate_id_list = torch.as_tensor(cate_id_list)
    subcate_id_list = torch.as_tensor(subcate_id_list)
    return tit_emb_list, abs_emb_list, cate_id_list, subcate_id_list


def getMindDataset(config, model, news_dict, glove_id, cate_dict, subcate_dict, sample_ratio=0):
    behaviors = pd.read_csv(config.DATA_PATH + "MIND" + config.DATASET_TYPE + "_" + model + "/behaviors.tsv", sep="\t",
                            names=config.BEHAVIORS_NAME)
    user_input_list = []

    index = [0]
    for behavior in behaviors.iterrows():
        history = behavior[1].get("History")
        if pd.isna(history):
            continue
        history = history.split(' ')
        if config.HISTORY_MIN_LENGTH > len(history):
            continue
        impressions = behavior[1]["Impressions"].split(' ')
        history_list = getHistorySampleByAfter(history, config.HISTORY_MAX_LENGTH)
        his_tit, his_abs, his_cat, his_subcat = getHisEmbByGlove(history_list, news_dict,
                                                                 glove_id,
                                                                 cate_dict, subcate_dict,
                                                                 config.FEATURE_MAX_LENGTH)

        if model == "train":
            imp_list, imp_label = getImpressionOnTrain(impressions, sample_ratio)
            n_imp = len(imp_list)
            for i in range(n_imp):
                imp_tit, imp_abs, imp_cat, imp_subcat = getHisEmbByGlove(imp_list[i],
                                                                         news_dict, glove_id,
                                                                         cate_dict, subcate_dict,
                                                                         config.FEATURE_MAX_LENGTH)
                temp = [his_tit, his_abs, his_cat, his_subcat, imp_tit, imp_abs, imp_cat, imp_subcat,
                        torch.as_tensor(imp_label[i], dtype=torch.float32)]
                user_input_list.append(temp)
        else:
            imp_list, imp_label = getImpressionOnTest(impressions)
            imp_tit_list, imp_abs_list, imp_cat_list, imp_subcat_list = getHisEmbByGlove(imp_list,
                                                                                         news_dict, glove_id,
                                                                                         cate_dict, subcate_dict,
                                                                                         config.FEATURE_MAX_LENGTH)
            for j in range(len(imp_list)):
                temp = []
                temp.append(his_tit)
                temp.append(his_abs)
                temp.append(his_cat)
                temp.append(his_subcat)
                temp.append(torch.unsqueeze(imp_tit_list[j], dim=0))
                temp.append(torch.unsqueeze(imp_abs_list[j], dim=0))
                temp.append(torch.as_tensor([imp_cat_list[j]]))
                temp.append(torch.as_tensor([imp_subcat_list[j]]))
                temp.append(torch.as_tensor(imp_label[j], dtype=torch.float32))
                user_input_list.append(temp)
            index.append(index[-1] + len(imp_list))
    print(model, len(user_input_list))
    return user_input_list, index


def NegSample(neg_impression, ratio):
    if ratio > len(neg_impression):
        return random.sample(neg_impression * (ratio // len(neg_impression) + 1), k=ratio)
    else:
        return random.sample(neg_impression, k=ratio)


def getImpressionOnTrain(impressions, neg_max_length):
    pos_impression = []
    neg_impression = []
    imp_list = []
    imp_label = []
    impression_dict = {}
    for impression in impressions:
        new_id, clk = impression.split("-")
        impression_dict[new_id] = int(clk)
        if float(clk) == 0:
            neg_impression.append(new_id)
        else:
            pos_impression.append(new_id)
    for pos_imp_id in pos_impression:
        temp_id = NegSample(neg_impression, neg_max_length)
        temp_id.append(pos_imp_id)
        random.shuffle(temp_id)
        tmp_label = [impression_dict[id] for id in temp_id]
        imp_list.append(temp_id)
        imp_label.append(tmp_label)
    return imp_list, imp_label


def getImpressionOnTest(impressions):
    imp_list = []
    imp_label = []
    for impression in impressions:
        new_id, clk = impression.split("-")
        imp_list.append(new_id)
        imp_label.append(float(clk))
    return imp_list, imp_label


def getHistorySampleByAfter(history, his_max_length):
    history = ["EMP"] * (his_max_length - len(history)) + history[: his_max_length]
    return history



if __name__ == '__main__':
    config = Config()
    getWordDictByGlove(config)
    getCategoryAndNewsDict(config)
