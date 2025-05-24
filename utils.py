import pandas as pd
from collections import Counter
import numpy as np
import json

with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

UNK = '<unk>'
PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'

data_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/All_Data/text_VA_clean.csv'
save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/All_Data/'

def preprocess_text_data(data_path):

    text_data = pd.read_csv(data_path)

    sentences = text_data['sentence'].apply(lambda x: x.lower().split()).tolist()
    
    counter = Counter(word for sentence in sentences for word in sentence)
    vocab = [PAD, UNK, SOS, EOS] + [word for word, _ in counter.most_common(args["vocab_size"]-4)]

    vocab = {word: index for index, word in enumerate(vocab)}
    sentence_lst = np.array([[vocab.get(word, vocab[UNK]) for word in sentence] + [vocab[EOS]] for sentence in sentences], dtype=object)
    vocab_lst = np.array(vocab, dtype=object)

    id2word = {v: k for k, v in vocab.items()}
    np.save(save_path+'id2word_text.npy', id2word)
        
    sos_index = args["SOS_INDEX"]
    eos_index = args["EOS_INDEX"]
    sentence_lst = [[sos_index] + list(sentence) + [eos_index] for sentence in sentence_lst]
    sentence_lst = np.array(sentence_lst, dtype=object)

    np.save(save_path+'vocab_text.npy', vocab_lst)
    np.save(save_path+'sentences_text.npy', sentence_lst)

    
    return sentence_lst, vocab_lst


def ids_to_words(ids, id2word):
    return [id2word.get(i, "<UNK>") for i in ids]

if __name__ == "__main__":
    preprocess_text_data(data_path)
