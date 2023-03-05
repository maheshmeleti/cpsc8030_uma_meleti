import torch
from torch.autograd import Variable


device = 0

def calculate_loss(loss_fn, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] -1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss/batch_size

    return loss

def train(model, epoch, loss_fn, parameters, optimizer, train_loader):
    model.train()
    print(epoch)

    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx)
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = Variable(avi_feats).to(device), Variable(ground_truths).to(device)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)
        print('Batch - ', batch_idx, ' Loss - ', loss)
        loss.backward()
        optimizer.step()

    loss = loss.item()
    return loss


