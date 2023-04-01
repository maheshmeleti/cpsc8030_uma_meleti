import requests
import json
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers
from transformers import BertModel, BertTokenizerFast, AdamW
# AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer, BertForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data_path = 'Spoken-SQuAD-master/spoken_train-v1.1.json'
test_data_path = 'Spoken-SQuAD-master/spoken_test-v1.1.json'
fig_save_path = 'figs'

def get_data(path): 
    #read each file and retrieve the contexts, qustions and answers
    with open(path, 'rb') as f:
        raw_data = json.load(f)
    contexts = []
    questions = []
    answers = []

    for group in raw_data['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    return contexts, questions, answers


train_contexts, train_questions, train_answers = get_data(train_data_path)
valid_contexts, valid_questions, valid_answers = get_data(test_data_path)


def add_answer_end(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

add_answer_end(train_answers, train_contexts)
add_answer_end(valid_answers, valid_contexts)

# Graphs
# token_lens = []

# for txt in train_contexts:
#     txt = txt.strip()  # remove leading and trailing whitespaces
#     token_lens.append(len(txt.split(' ')))

# plt.hist(token_lens,  bins=20)  # density=False would make counts
# plt.ylabel('Count')
# plt.xlabel('Length')
# plt.title('Distribution of Train Context Lengths')
# plt.savefig(os.path.join(fig_save_path, 'Distribution_train.png'))


# ans_lens = []
# for txt in train_questions:
#     txt = txt.strip()  # remove leading and trailing whitespaces
#     ans_lens.append(len(txt.split(' ')))


# plt.hist(ans_lens,  bins=20)  # density=False would make counts
# plt.ylabel('Count')
# plt.xlabel('Length')
# plt.title('Distribution of Train Question Lengths')
# plt.savefig(os.path.join(fig_save_path, 'Distribution_train_question.png'))

MAX_LENGTH = 512
MODEL_PATH = "bert-base-uncased"

tokenizerFast = BertTokenizerFast.from_pretrained(MODEL_PATH)

doc_stride = 128
train_encodings = tokenizerFast(train_questions, train_contexts,  max_length = MAX_LENGTH,truncation=True,stride=doc_stride,
        padding=True)
valid_encodings = tokenizerFast(valid_questions,valid_contexts,  max_length = MAX_LENGTH, truncation=True,stride=doc_stride,
        padding=True)

def start_and_end(idx):
    ret_start = 0
    ret_end = 0
    answer_encoding_fast = tokenizerFast(train_answers[idx]['text'],  max_length = MAX_LENGTH, truncation=True, padding=True)
    for a in range( len(train_encodings['input_ids'][idx]) -  len(answer_encoding_fast['input_ids']) ): #len(train_encodings['input_ids'][0])):
        match = True
        for i in range(1,len(answer_encoding_fast['input_ids']) - 1):
            if (answer_encoding_fast['input_ids'][i] != train_encodings['input_ids'][idx][a + i]):
                match = False
                break
            if match:
                ret_start = a+1
                ret_end = a+i+1
                break
    return(ret_start, ret_end)

start_positions = []
end_positions = []
ctr = 0
for h in range(len(train_encodings['input_ids'])):
    s, e = start_and_end(h)
    start_positions.append(s)
    end_positions.append(e)
    if s==0:
        ctr = ctr + 1

train_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
valid_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SQAD_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][i]),
            'token_type_ids': torch.tensor(self.encodings['token_type_ids'][i]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][i]),
            'start_positions': torch.tensor(self.encodings['start_positions'][i]),
            'end_positions': torch.tensor(self.encodings['end_positions'][i])
        }
    def __len__(self):
        return len(self.encodings['input_ids'])
    
train_dataset = SQAD_Dataset(train_encodings)
valid_dataset = SQAD_Dataset(valid_encodings)

train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=1)

bert_model = BertModel.from_pretrained(MODEL_PATH)

class QA_Lang_Model(nn.Module):
    def __init__(self):
        super(QA_Lang_Model, self).__init__()
        self.bert = bert_model
        self.drop_out = nn.Dropout(0.1)
        self.l1 = nn.Linear(768 * 2, 768 * 2)
        self.l2 = nn.Linear(768 * 2, 2)
        self.linear_relu_stack = nn.Sequential(
            self.drop_out,
            self.l1,
            nn.LeakyReLU(),
            self.l2 
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        model_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = model_output[2]
        out = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)  # taking Start logits from last BERT layer, End Logits from third to last layer
        logits = self.linear_relu_stack(out)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

model = QA_Lang_Model()

def focal_loss(start_logits, end_logits, start_positions, end_positions, gamma):
    
    #calculate Probabilities by applying Softmax to the Start and End Logits. Then get 1 - probabilities
    smax = nn.Softmax(dim=1)
    probs_start = smax(start_logits)
    inv_probs_start = 1 - probs_start
    probs_end = smax(end_logits)
    inv_probs_end = 1 - probs_end
    
    #get log of probabilities. Note: NLLLoss required log probabilities. This is the Natural Log (Log base e)
    lsmax = nn.LogSoftmax(dim=1)
    log_probs_start = lsmax(start_logits)
    log_probs_end = lsmax(end_logits)
    
    nll = nn.NLLLoss()
    
    fl_start = nll(torch.pow(inv_probs_start, gamma)* log_probs_start, start_positions)
    fl_end = nll(torch.pow(inv_probs_end, gamma)*log_probs_end, end_positions)
    
    #return mean of the Loss for the start and end logits
    return ((fl_start + fl_end)/2)

optim = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)
total_acc = []
total_loss = []

def train_(model, dataloader, epoch):
    model = model.train()
    losses = []
    acc = []
    tracking_checkpoint = 0
    for batch in tqdm(dataloader, desc = 'Running Epoch '):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        out_start, out_end = model(input_ids=input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
        loss = focal_loss(out_start, out_end, start_positions, end_positions,1) #using gamma = 1
        losses.append(loss.item())
        loss.backward()
        optim.step()
        
        start_pred = torch.argmax(out_start, dim=1)
        end_pred = torch.argmax(out_end, dim=1)
            
        acc.append(((start_pred == start_positions).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_positions).sum()/len(end_pred)).item())
        tracking_checkpoint = tracking_checkpoint + 1
        if tracking_checkpoint==250 and epoch==1:
            total_acc.append(sum(acc)/len(acc))
            loss_avg = sum(losses)/len(losses)
            total_loss.append(loss_avg)
            tracking_checkpoint = 0
    ret_acc = sum(acc)/len(acc)
    ret_loss = sum(losses)/len(losses)
    return(ret_acc, ret_loss)


def infer_model(model, dataloader):
    model = model.eval()
    losses = []
    acc = []
    ctr = 0
    answer_list=[]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc = 'Running Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            
            out_start, out_end = model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
            # print("out_start",out_start.shape)
            start_pred = torch.argmax(out_start)
            end_pred = torch.argmax(out_end)
            answer = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(input_ids[0][start_pred:end_pred]))
            tanswer = tokenizerFast.convert_tokens_to_string(tokenizerFast.convert_ids_to_tokens(input_ids[0][start_true[0]:end_true[0]]))
            answer_list.append([answer,tanswer])
        #ret_loss = sum(losses)/len(losses)
    return answer_list

from evaluate import load
wer = load("wer")
EPOCHS = 6
model.to(device)
wer_list=[]
for epoch in range(EPOCHS):
    print('Epoch - {}'.format(epoch))
    train_acc, train_loss = train_(model, train_data_loader, epoch+1)
    print(f"Train Accuracy: {train_acc}      Train Loss: {train_loss}")
    answer_list = infer_model(model, valid_data_loader)
    pred_answers=[]
    true_answers=[]
    for i in range(len(answer_list)):
        if(len(answer_list[i][0])==0):
            answer_list[i][0]="$"
        if(len(answer_list[i][1])==0):
            answer_list[i][1]="$"
        pred_answers.append(answer_list[i][0])
        true_answers.append(answer_list[i][1])
    wer_score = wer.compute(predictions=pred_answers, references=true_answers)
    wer_list.append(wer_score)

print('WER- ',wer_list)
