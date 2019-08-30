"""
The script expects to find a .pkl and a .pt file in the 'models' folder.
To run, the parser requires a -model option.
The script expects both model and dictionary to follow the following naming convention.
'name_model.pt'
'name_dict.pkl'
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, required=True)
parser.add_argument('-length', type=int, default=40)
parser.add_argument('-rough', type=bool, default=False)
parser.add_argument('-capitalization', type=bool, default=False)
args = parser.parse_args()

from charvocabulary import charVocabulary
from charmodel import charModel
import torch
import pickle
import os

def generate_name(modelname, rough=False, capitalization=False):

    path = os.path.join('models', modelname)
    dict_path = path+'_dict.pkl'
    model_path = path+'_model.pt'

    token_to_idx = pickle.load(open(dict_path,'rb'))
    vocab = charVocabulary(token_to_idx=token_to_idx)

    model = charModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    beginid = vocab.begin_idx
    begintensor = torch.tensor([beginid], dtype=torch.int64).unsqueeze(dim=0)
    ind = [begintensor]
    t = 1
    x_t = ind[1-1]
    h_t = None

    for t in range(1,args.length+1):
        x_t = ind[t-1]
        emb_t = model.emb(x_t)
        rnn_t, h_t = model.rnn(emb_t, h_t)
        pred_vector = model.fc(rnn_t.squeeze(dim=1))
        prob_vector = torch.nn.functional.softmax(pred_vector, dim=1)
        winner = torch.multinomial(prob_vector, num_samples=1)
        ind.append(winner)

    s = ""
    for i in range(len(ind)):
        idx = ind[i].item()
        s += vocab.lookup_idx(idx)
        
    if args.rough:
        return s
    else:
        i = 0
        while s[i] != '>' and i < len(s):
            i+=1

        out = ""
        j = i+1
        while s[j] != '<' and j < len(s):
            out += s[j]
            j+=1
        
        if args.capitalization:
            out = out.capitalize()
        return out

print(generate_name(modelname=args.model, rough=args.rough, capitalization=args.capitalization))