import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json
import numpy as np

with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/Saves/'

def get_text_data():
    '''
    get the text data from the saved numpy arrays and put them in TextDataset class.
    '''
    sentence_lst = np.load(save_path + 'Text_VAE_sentence_lst.npy',allow_pickle=True)
    label_lst = np.load(save_path + 'Text_VAE_label_lst.npy',allow_pickle=True)    
    
    return sentence_lst, label_lst

class TextDataset(Dataset):
    def __init__(self, sentence, label, max_len=50, mode="train"):
        self.sentence = np.array(sentence)
        self.label = np.array(label)
        self.max_len = max_len

        random_state = 20
        test_ratio = 0.1

        sentence_train, sentence_test, label_train, label_test = train_test_split(
            self.sentence, self.label, test_size=test_ratio, random_state=random_state
        )

        if mode == "train":
            self.sentence, self.label = sentence_train, label_train
        elif mode == "val":
            self.sentence, self.label = sentence_test, label_test

        self.label = torch.tensor(self.label, dtype=torch.long)

        # Pad or truncate sentences
        sentences = []
        for sentence in self.sentence:
            if len(sentence) > self.max_len:
                sentence = sentence[:self.max_len]
            else:
                sentence = np.pad(sentence, (0, self.max_len - len(sentence)), mode='constant', constant_values=args["PAD_INDEX"])
            sentences.append(torch.tensor(sentence, dtype=torch.long))
        self.sentence = torch.stack(sentences)
        
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        label = self.label[idx]
        return sentence, label
    
if __name__ == "__main__":
    sentence_lst, label_lst = get_text_data()
    print("Sample sentence:", sentence_lst[0])
    print("Sample label:", label_lst[0])

    train_dataset = TextDataset(sentence_lst, label_lst, mode="train")
    test_dataset = TextDataset(sentence_lst, label_lst, mode="val")

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
