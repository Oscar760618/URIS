import numpy as np
import config
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, mode="train"):

        # alae
        # img = np.load(config.IMAGE_DATA_ROOT, allow_pickle=True).item()
        # self.img = img
        # vqvae/bvae
        
        text = torch.load(config.TEXT_DATA_ROOT)
        self.text = text

        muse = np.load(config.MUSE_ROOT,allow_pickle=True)
        self.muse = torch.from_numpy(muse)

        if mode == "train":
            text_pos_pairs = open(config.TRAIN_TEXT_POS_ROOT,"r",encoding='utf-8')
            self.text_pos_pairs = text_pos_pairs.readlines()

            text_neg_pairs = open(config.TRAIN_TEXT_NEG_ROOT,"r",encoding='utf-8')
            self.text_neg_pairs = text_neg_pairs.readlines()


            muse_pos_pairs = open(config.TRAIN_MUSE_POS_ROOT,"r",encoding='utf-8')
            self.muse_pos_pairs = muse_pos_pairs.readlines()

            muse_neg_pairs = open(config.TRAIN_MUSE_NEG_ROOT, "r",encoding='utf-8')
            self.muse_neg_pairs = muse_neg_pairs.readlines()

        elif mode == "test":
            text_pos_pairs = open(config.TEST_TEXT_POS_ROOT,"r",encoding='utf-8')
            self.text_pos_pairs = text_pos_pairs.readlines()

            text_neg_pairs = open(config.TEST_TEXT_NEG_ROOT,"r",encoding='utf-8')
            self.text_neg_pairs = text_neg_pairs.readlines()


            muse_pos_pairs = open(config.TEST_MUSE_POS_ROOT,"r",encoding='utf-8')
            self.muse_pos_pairs = muse_pos_pairs.readlines()

            muse_neg_pairs = open(config.TEST_MUSE_NEG_ROOT, "r",encoding='utf-8')
            self.muse_neg_pairs = muse_neg_pairs.readlines()


    def __len__(self):
        return len(self.muse_pos_pairs)


    def __getitem__(self, index):

        text_num_int = self.text_pos_pairs[index].strip().split(" ")[0]          
        text_num_int = int(text_num_int) + 1
        
        pos_text_num = self.text_pos_pairs[index].strip().split(" ")[1:]
        pos_text = torch.cat([self.text[int(i)].unsqueeze(0) for i in pos_text_num], dim=0)

        neg_text_num = self.text_neg_pairs[index].strip().split(" ")[1:]
        neg_text = torch.cat([self.text[int(i)].unsqueeze(0) for i in neg_text_num], dim=0)

        # inter-model
        pos_muse_num = self.muse_pos_pairs[index].strip().split(" ")[1:]
        pos_muse = torch.cat([self.muse[int(i)].unsqueeze(0) for i in pos_muse_num], dim=0)
        # print(pos_muse.shape)

        neg_muse_num = self.muse_neg_pairs[index].strip().split(" ")[1:]
        neg_muse = torch.cat([self.muse[int(i)].unsqueeze(0) for i in neg_muse_num], dim=0)
        # print(neg_muse.shape)


        return text_num_int, pos_text, neg_text, pos_muse, neg_muse


if __name__ == "__main__":
    dataset = MapDataset(mode="train")
    # print(len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)



    for i, (text_num_int, pos_text, neg_text, pos_muse, neg_muse) in enumerate(loader):
        print(text_num_int)
        print(pos_text.shape)
        print(neg_text.shape)
        print(pos_muse.shape)
        print(neg_muse.shape)
        break
        # a = 1
