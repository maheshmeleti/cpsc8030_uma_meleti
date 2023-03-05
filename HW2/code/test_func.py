import sys
import torch
import json
from main import test_data, test, MODELS
from decoder import Decoder_lstm_attn 
from encoder import Encoder_lstm
from attention import attention 
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

model = torch.load('./saved_model/model_final.h5', map_location=lambda storage, loc: storage)
# filepath = r'C:\Users\UMA\Desktop\grad\Deep_Learning\code\HW2\MLDS_hw2_1_data\MLDS_hw2_1_data\training_data\feat'
print('Arg 1 - ', model)


dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True)

with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

# model = model.cuda()
ss = test(testing_loader, model, i2w)

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))


test = json.load(open(r"C:\Users\UMA\Desktop\grad\Deep_Learning\code\HW2\MLDS_hw2_1_data\MLDS_hw2_1_data\testing_label.json"))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
#count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))

