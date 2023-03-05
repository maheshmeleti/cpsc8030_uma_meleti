import os
import numpy as np
import re
import json
import torch
from torch.utils.data import DataLoader, Dataset

def data_preprocess(filepath):
    with open(os.path.join(filepath + '/training_label.json'), 'r') as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            word_sentence = re.sub('[.!,;?]', ' ', s).split()
            for word in word_sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    word_dict = {}
    for word in word_count:
        if word_count[word] > 4:
            word_dict[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, word_dict

def s_split(sentence, word_dict, w2i):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            sentence[i] = 3
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence

def annotate(data_root, label_file, word_dict, w2i):
    label_json = os.path.join(data_root, label_file)
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = s_split(s, word_dict, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption

def avi(data_root, files_dir):
    avi_data = {}
    training_feats = data_root + files_dir
    files = os.listdir(training_feats)
    i = 0
    for file in files:
        print(i)
        i+=1
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

class training_data(Dataset):
    def __init__(self, data_root, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = avi(data_root, label_file)
        self.w2i = w2i
        self.data_pair = annotate(data_root, files_dir, word_dict, w2i)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000.
        return torch.Tensor(data), torch.Tensor(sentence)

def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths