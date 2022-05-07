"""
@author:32369
@file:train.py
@time:2021/11/08
"""
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from CMM_CT_M import CMM_CT_M
from Config import Config
from ProcessMindByGlove import getMindDataset
from dataloader import MindDataset
from metric import validateStep


def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, device, lr, glove_dict, cate_length, subcate_length, train_loader, dev_loader, index):
    # model = CMM_T(config, glove_dict, cate_length, subcate_length).to(device)
    model = CMM_CT_M(config, glove_dict, cate_length, subcate_length).to(device)
    # model = CMM_CTA(config, glove_dict, cate_length, subcate_length).to(device)
    # model = CMM_TA(config, glove_dict, cate_length, subcate_length).to(device)

    loss_fcn = nn.BCELoss()  # Loss函数
    loss_fcn = loss_fcn.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1, 2, 3, 5], gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    best_auc = 0
    for _ in range(config.EPOCHS):
        """训练部分"""
        model.train()  # 启用droupt()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            htl, hal, hcl, hsl, itl, ial, icl, isl, label = x
            htl, hal, hcl, hsl, itl, ial, icl, isl, label = htl.to(device), hal.to(device), hcl.to(device), hsl.to(
                device), itl.to(device), ial.to(device), icl.to(device), isl.to(device), label.to(device)
            pred = model(htl, hal, hcl, hsl, itl, ial, icl, isl)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            if (idx + 1) % config.PRINT_LENGHT == 0 or (idx + 1) == len(train_loader):
                print("Times {} | Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} |Step Time {:.4f}".format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                    _ + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1),
                    (time.time() - start_time) / (idx + 1)))
        """推断部分"""
        model.eval()  # 不启用droupt()
        with torch.no_grad():
            preds, labels = [], []
            for idx, x in enumerate(dev_loader):
                htl, hal, hcl, hsl, itl, ial, icl, isl, label = x
                htl, hal, hcl, hsl, itl, ial, icl, isl, label = htl.to(device), hal.to(device), hcl.to(device), hsl.to(
                    device), itl.to(device), ial.to(device), icl.to(device), isl.to(device), label.to(device)
                score = model(htl, hal, hcl, hsl, itl, ial, icl, isl)
                score = score.reshape(-1).cpu().numpy().tolist()
                label = label.cpu().numpy().tolist()
                preds.extend(score)
                labels.extend(label)
            auc, mrr, ndcg5, ndcg10 = validateStep(labels, preds, index)
        if best_auc < np.mean(auc):
            best_auc = auc
            # torch.save(model.state_dict(), "./model")
        print('AUC: %.6f, MRR: %.6f, NDCG5: %.6f， NDCG10: %.6f' % (auc, mrr, ndcg5, ndcg10))
        scheduler.step()
    print("最高精度为：", best_auc)
    return best_auc


def ReapeatTrain(config, device, glove_dict, cate_length, subcate_length, train_loader, dev_loader, index, N=5):
    auc = []
    for i in range(N):
        best = train(config, device, config.LR, glove_dict, cate_length, subcate_length, train_loader, dev_loader,
                     index)
        auc.append(best)
    print("AUC精度：", auc)
    print("AUC平均精度：", np.mean(auc))


if __name__ == '__main__':
    config = Config()
    seed_torch()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    glove_dict = np.load(config.PROCESS_DATA_PATH + "glove_dict.npy", allow_pickle=True)
    cate2id = np.load(config.PROCESS_DATA_PATH + "cate2id.npy", allow_pickle=True).item()
    subcate2id = np.load(config.PROCESS_DATA_PATH + "subcate2id.npy", allow_pickle=True).item()
    news_dict = np.load(config.PROCESS_DATA_PATH + "news_dict.npy", allow_pickle=True).item()
    word2id = np.load(config.PROCESS_DATA_PATH + "word2id.npy", allow_pickle=True).item()

    train_sample, _ = getMindDataset(config, "train", news_dict, word2id, cate2id, subcate2id, config.nsample)
    test_sample, index = getMindDataset(config, "dev", news_dict, word2id, cate2id, subcate2id)

    train_dataset = MindDataset(train_sample)
    test_dataset = MindDataset(test_sample)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    dev_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 4, shuffle=False,
                            num_workers=config.NUM_WORKERS)

    train(config, device, config.LR, glove_dict, len(cate2id) + 1, len(subcate2id) + 1, train_loader, dev_loader, index)
    # ReapeatTrain(config, device, glove_dict, len_cate, len_subcate, train_loader, dev_loader, index, N=10)
