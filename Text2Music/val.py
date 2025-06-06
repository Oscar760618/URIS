from importlib.resources import path
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import pickle
import numpy as np

from dataset import MapDataset
from model import CLloss
import config
import io

SAVE_ROOT = r"D:\PolyU\URIS\Part2_projects\EIMG-main\EIMG-main\params\params_5_11.166.pt"
MUSIC_ROOT = r"D:\PolyU\URIS\Part2_projects\EIMG-main\EIMG-main\params"

def i2m(img_dim, save_path, muse_pwd_path):

    # init model
    clnet = CLloss(img_dim=img_dim, mus_dim=280)


    state = torch.load(save_path)
    clnet.load_state_dict(state['model_state_dict'])

    print("Loading params/", save_path)
    
    clnet.cuda()
    clnet.eval()


    # dataset
    dataset = MapDataset(mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # produce
    for idx, (text_num_int, pos_img, neg_img, pos_muse, neg_muse) in enumerate(loader):
           

        pos_img, neg_img, pos_muse, neg_muse = pos_img.cuda(), neg_img.cuda(), pos_muse.cuda(), neg_muse.cuda()
        
        with torch.no_grad():
            pro_muse = clnet.forward(pos_img, neg_img, pos_muse, neg_muse, training=False)
        pro_muse = pro_muse.unsqueeze(dim=0)

        # break
        torch.save(pro_muse, muse_pwd_path + "/" + f"{text_num_int.item()}" + ".pt")
    

        

if __name__ == "__main__":
    i2m(img_dim=512, save_path=SAVE_ROOT, muse_pwd_path=MUSIC_ROOT)