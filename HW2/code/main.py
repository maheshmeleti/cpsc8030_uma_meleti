import torch.nn as nn
import os
from data_processing import data_preprocess, training_data, minibatch
from encoder import Encoder_lstm
from decoder import Decoder_lstm_attn
from model import MODELS
import pickle
from torch.utils.data import DataLoader, Dataset
from training import train
import torch.optim as optim
import torch
from torch.autograd import Variable
import numpy as np

device = 0
def main(n_epochs, data_root):
    i2w, w2i, word_dict = data_preprocess(data_root)
    with open('i2w.pickle', 'wb') as handle:
        pickle.dump(i2w, handle, protocol = pickle.HIGHEST_PROTOCOL)
    label_file = '/training_data/feat'
    files_dir = 'training_label.json'
    train_dataset = training_data(data_root, label_file, files_dir, word_dict, w2i)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=minibatch)


    encoder = Encoder_lstm()
    decoder = Decoder_lstm_attn(512, len(i2w) + 4, len(i2w) + 4, 1024, 0.3)
    model = MODELS(encoder=encoder, decoder=decoder)

    model = model #.to(device)
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)
    loss_arr = []
    for epoch in range(n_epochs):
        loss = train(model, epoch + 1, loss_fn, parameters, optimizer, train_dataloader)
        loss_arr.append(loss)

    with open('SavedModel/loss_values.txt', 'w') as f:
        for item in loss_arr:
            f.write("%s\n" % item)
    torch.save(model, "{}/{}.h5".format('SavedModel', 'model0'))
    print("Training finished")

def test(test_loader, model, i2w):
    model.eval()
    ss = []
    
    for batch_idx, batch in enumerate(test_loader):
     
        id, avi_feats = batch
        avi_feats = avi_feats #.to(device)
        id, avi_feats = id, Variable(avi_feats).float()
        
        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions
        
        result = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(id, result)
        for r in rr:
            ss.append(r)
    return ss

class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
            
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]
    
if __name__ == "__main__":
    n_epochs = 10
    data_root = '../../umeleti/data/'
    data_root = r'C:\Users\UMA\Desktop\grad\Deep_Learning\code\HW2\MLDS_hw2_1_data\MLDS_hw2_1_data'
    main(n_epochs, data_root)