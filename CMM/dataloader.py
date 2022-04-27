from torch.utils.data import Dataset


class MindDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        line = self.data[index]
        his_tit = line[0]
        his_abs = line[1]
        his_cate = line[2]
        his_subcate = line[3]

        imp_tit = line[4]
        imp_abs = line[5]
        imp_cate = line[6]
        imp_subcate = line[7]

        label = line[8]

        return his_tit, his_abs, his_cate, his_subcate, imp_tit, imp_abs, imp_cate, imp_subcate, label

    def __len__(self):
        return len(self.data)
