from charvocabulary import charVocabulary
from charvectorizer import charVectorizer
from chardataset import charDataset
from charmodel import charModel
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
# source = source text (already cleaned!)
parser.add_argument('-source', type=str, required=True)
# ne = number of epochs
parser.add_argument('-ne', type=int, default=88)
# dropout speeds up training by turning off neurons
parser.add_argument('-dropout', type=float, default=None)
# torch device
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-device', type=str, default='cuda')

args = parser.parse_args()

vocab = charVocabulary()
txt_path = os.path.join('source', args.source+'.txt')
vocab.add_txt(txt_path)

dict_path = os.path.join('source', args.source+'_dict.pkl')
pickle.dump(vocab.token_to_idx, open(dict_path,'wb'))

maskid = vocab.mask_idx

vectorizer = charVectorizer(vocab=vocab)

corpus = pd.read_csv(txt_path, header=None).dropna().reset_index()[0]

if args.dropout:
    model = charModel(vocab_size=len(vocab),padding_idx=maskid,dropout_p=args.dropout)
else:
    model = charModel(vocab_size=len(vocab),padding_idx=maskid)

ds = charDataset(vectorizer=vectorizer, corpus=corpus)
dl = DataLoader(ds, batch_size=4, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

from charsample import generate_sample
print(generate_sample(model=model,vectorizer=vectorizer))

# set device
if args.cuda and torch.cuda.is_available():
    device = torch.device(args.device)
else:
    device = torch.device('cpu')
# transfer model to device
model.to(device)
print(f"\nDevice used is {device}.")

num_epochs = args.ne
try:
    for epoch in range(num_epochs):
        print(f"\nEpoch number {epoch+1} is starting now.")
        model.train()
        for x,y in tqdm.tqdm(dl):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            batch_size, seq_len, feats = y_pred.shape
            y_pred_loss = y_pred.view(batch_size*seq_len,feats)
            y_loss = y.view(-1)

            loss = F.cross_entropy(y_pred_loss, y_loss, ignore_index=maskid)
            loss.backward()
            optimizer.step()

        model.eval()

        model.to('cpu')
        for i in range(5):
            print(generate_sample(model=model,vectorizer=vectorizer))
        model.to(device)

        print(f"Epoch number {epoch+1} has now concluded.")

except KeyboardInterrupt:
    print("\nTraining was interrupted. That's ok, I'll still save the latest model.")

model_path = os.path.join('models', args.source+'_model.pt')
torch.save(model.state_dict(), model_path)