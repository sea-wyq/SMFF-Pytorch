import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim_in, dim_k, dim_v, num_heads):
        super().__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
        norm_fact = 1 / torch.sqrt(torch.tensor(self.dim_k // nh))
        dist = torch.matmul(q, k.transpose(2, 3))  # batch, nh, n, n
        dist = torch.mul(dist, norm_fact)  # batch, nh, n, n
        dist = torch.softmax(dist, dim=3)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


class NewEncoder(nn.Module):
    def __init__(self, config, Q_dim, K_dim, V_dim, Heads):
        super().__init__()
        self.droupt = nn.Dropout(config.DROUPT)
        self.attention = MultiHeadSelfAttention(Q_dim, K_dim, V_dim, Heads)
        self.news_layer = nn.Sequential(nn.Linear(V_dim, config.HIDDEN_SIZE),
                                        nn.Tanh(),  # 20 12 64
                                        nn.Linear(config.HIDDEN_SIZE, 1),
                                        nn.Flatten(), nn.Softmax(dim=1))  # 20 12

    def forward(self, title_emb_input):
        word_att = self.droupt(self.attention(self.droupt(title_emb_input)))  # 64 12 128
        attention_weight = self.news_layer(word_att)  # 64 12
        attention_weight = torch.unsqueeze(attention_weight, dim=2)  # 64 12 1
        new_emb = torch.sum(word_att * attention_weight, dim=1)  # 64 128
        return new_emb


class UserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.new_encoder = NewEncoder(config, config.Q, config.K, config.V, config.NUM_HEADS)
        self.user_encoder = NewEncoder(config, config.V, config.V, config.V, config.NUM_HEADS)
        self.wt = nn.Parameter(torch.rand((1,)))
        self.wa = nn.Parameter(torch.rand((1,)))

    def forward(self, his_tit_emb, imp_tit_emb, his_abs_emb, imp_abs_emb):
        his_tit_emb, imp_tit_emb = torch.transpose(his_tit_emb, 0, 1), torch.transpose(imp_tit_emb, 0, 1)
        his_abs_emb, imp_abs_emb = torch.transpose(his_abs_emb, 0, 1), torch.transpose(imp_abs_emb, 0, 1)

        tit_emb = torch.stack(
            [self.new_encoder(his_tit_emb[i]) for i in range(his_tit_emb.shape[0])], dim=1)

        abs_emb = torch.stack(
            [self.new_encoder(his_abs_emb[i]) for i in range(his_abs_emb.shape[0])], dim=1)

        user_emb = torch.add(self.wt * tit_emb, self.wa * abs_emb)
        user = self.user_encoder(user_emb)

        score = []
        for i in range(imp_tit_emb.shape[0]):
            imp = torch.add(self.wt * self.new_encoder(imp_tit_emb[i]),
                            self.wa * self.new_encoder(imp_abs_emb[i]))
            score.append(torch.sum(user * imp, dim=1))
        score = torch.stack(score, dim=1)  # 64 10
        return score


class CMM(nn.Module):
    def __init__(self, config, glove_dict, cate_size, subcate_size):
        super().__init__()
        self.cate_length = config.FEATURE_EMB_LENGTH // 2
        self.glove_embdding = nn.Embedding.from_pretrained(torch.as_tensor(glove_dict, dtype=torch.float32))
        self.cate_embdding = nn.Embedding(cate_size, self.cate_length)
        self.subcate_embdding = nn.Embedding(subcate_size, self.cate_length)
        for p in self.parameters():
            p.requires_grad = False
        self.user_encoder = UserEncoder(config)

    def forward(self, his_tit, his_abs, his_cate, his_subcate, imp_tit, imp_abs, imp_cate, imp_subcate):
        his_tit_emb, his_abs_emb = self.glove_embdding(his_tit), self.glove_embdding(his_abs)
        imp_tit_emb, imp_abs_emb = self.glove_embdding(imp_tit), self.glove_embdding(imp_abs)

        his_cate_emb = torch.cat((self.cate_embdding(his_cate), self.subcate_embdding(his_subcate)), dim=2)
        imp_cate_emb = torch.cat((self.cate_embdding(imp_cate), self.subcate_embdding(imp_subcate)), dim=2)

        his_cate_emb = torch.unsqueeze(his_cate_emb, dim=2)
        imp_cate_emb = torch.unsqueeze(imp_cate_emb, dim=2)

        input1 = torch.cat((his_cate_emb, his_tit_emb), dim=2)
        input2 = torch.cat((imp_cate_emb, imp_tit_emb), dim=2)
        input3 = torch.cat((his_cate_emb, his_abs_emb), dim=2)
        input4 = torch.cat((imp_cate_emb, imp_abs_emb), dim=2)

        tit_score = self.user_encoder(input1, input2, input3, input4)
        score = torch.sigmoid(tit_score)
        return score
