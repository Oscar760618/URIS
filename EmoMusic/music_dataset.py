'''
code for getting data from saved numpy arrays and the dataset class for the VGMIDI dataset
'''
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

VA_data_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/Text_dataset/labels.csv'
midi_data_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/Mus_dataset'
save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/Saves/'
path = 'D:/PolyU/URIS/Part2_projects/EIMG-main/EIMG-main/Dataset/dataset/'

def get_vgmidi():
    '''
    get the MIDI data from the saved numpy arrays and put them in VGMIDIDataset class.
    '''
    data_lst = np.load(save_path + "Music_VAE_data_lst.npy", allow_pickle=True)
    rhythm_lst = np.load(save_path + "Music_VAE_rhythm_lst.npy", allow_pickle=True)
    note_density_lst = np.load(save_path + "Music_VAE_note_density_lst.npy", allow_pickle=True)
    chroma_lst = np.load(save_path + "Music_VAE_chroma_lst.npy", allow_pickle=True)

    # allow_pickle：这是一个布尔参数，用于控制是否允许使用 Python 的 pickle 模块来序列化和反序列化对象。
    # True：允许使用 pickle。这意味着你可以保存和加载包含非基本数据类型（如自定义对象、列表等）的 NumPy 数组。
    # False（默认值）：不允许使用 pickle。在这种情况下，只有基本的 NumPy 数据类型（如整数、浮点数等）可以被保存和加载。

    valence_lst = np.load(save_path + "Music_VAE_valence_lst.npy")
    arousal_lst = np.load(save_path + "Music_VAE_arousal_lst.npy")
    label_lst = np.load(save_path + "Music_VAE_label_lst.npy")

    
    print("Shapes for: Data, Rhythm Density, Note Density, Chroma")
    print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)
    print("Shapes for: Arousal, Valence, Label")
    print(arousal_lst.shape, valence_lst.shape, label_lst.shape)
    #return data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst
    return data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst, label_lst
    
class VGMIDIDataset(Dataset):
    '''
    VGMIDI dataset loader.
    '''
    def __init__(self, data, rhythm, note, chroma, arousal, valence, label, mode="train"):
        super().__init__()
        # inputs = data, rhythm, note, chroma, arousal, valence, label
        indexed = []

        # train test split
        random_state = 20
        test_ratio = 0.1

        # list_dataset = data, rhythm, note, chroma, arousal, valence
        data_train, data_test, rhythm_train, rhythm_test,\
        note_train, note_test, chroma_train, chroma_test,\
        arousal_train, arousal_test, valence_train, valence_test,\
        label_train, label_test = train_test_split(data, rhythm, note, chroma, arousal, valence, label, test_size=test_ratio, random_state=random_state)

        train_data = data_train, rhythm_train, note_train, chroma_train, arousal_train, valence_train, label_train
        test_data = data_test, rhythm_test, note_test, chroma_test, arousal_test, valence_test, label_test

        if mode == "train":
            indexed = train_data
        elif mode == "val":
            indexed = test_data
        


        self.data, self.rhythm, self.note, self.chroma, self.arousal, self.valence, self.label= indexed
        self.data = [torch.Tensor(np.insert(k, -1, 1)) for k in self.data]
        self.data = torch.nn.utils.rnn.pad_sequence(self.data, batch_first=True)

        # put this before applying torch.Tensor
        self.r_density = [Counter(list(k))[1] / len(k) for k in self.rhythm]
        self.n_density = np.array([sum(k) / len(k) for k in self.note])

        self.rhythm = [torch.Tensor(k) for k in self.rhythm]
        self.note = [torch.Tensor(k) for k in self.note]

        self.rhythm = torch.nn.utils.rnn.pad_sequence(self.rhythm, batch_first=True)
        self.note = torch.nn.utils.rnn.pad_sequence(self.note, batch_first=True)
        self.chroma = torch.nn.utils.rnn.pad_sequence(self.rhythm, batch_first=True)

        ###padding###
        target_length = 448
        r_padding_length = target_length - self.rhythm.size(1)
        n_padding_length = target_length - self.note.size(1)
        c_padding_length = target_length - self.chroma.size(1)
        self.rhythm = F.pad(self.rhythm, (0, r_padding_length))
        self.note = F.pad(self.note, (0, n_padding_length))
        self.chroma = F.pad(self.chroma, (0, c_padding_length))
        #############

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        n = self.note[idx]
        c = self.chroma[idx]
        a = self.arousal[idx]
        v = self.valence[idx]
        l = self.label[idx]
        
        r_density = self.r_density[idx]
        n_density = self.n_density[idx]
        
        return x, r, n, c, a, v, l, r_density, n_density
    
if __name__ == '__main__':
    get_vgmidi()
    