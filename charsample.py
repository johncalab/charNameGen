import torch
import torch.nn.functional as F

def generate_sample(
                model,
                vectorizer,
                sample_size=30,
                rough=True,
                capitalization=False,
                cuda=False,
                device=torch.device('cpu')):
    
    beginid = vectorizer.vocab.begin_idx
    begintensor = torch.tensor([beginid], dtype=torch.int64).unsqueeze(dim=0).to(device)
    ind = [begintensor]
    t = 1
    x_t = ind[1-1]
    h_t = None

    sample_size=20
    for t in range(1,sample_size+1):
        x_t = ind[t-1]
        emb_t = model.emb(x_t)
        rnn_t, h_t = model.rnn(emb_t, h_t)
        pred_vector = model.fc(rnn_t.squeeze(dim=1))
        # this squeezing is equivalent to the reshaping procedure in the model itself
        # this is due to batch_len = 1
        prob_vector = F.softmax(pred_vector, dim=1)
        winner = torch.multinomial(prob_vector, num_samples=1)
        ind.append(winner)

    s = ""
    for i in range(len(ind)):
        idx = ind[i].item()
        s += vectorizer.vocab.lookup_idx(idx)
    
    if rough:
        return s
    else:
        i = 0
        while i < len(s) and s[i] != '>':
            i+=1

        out = ""
        j = i+1
        while j < len(s) and s[j] != '<':
            out += s[j]
            j+=1
        
        if capitalization:
            out = out.capitalize()

        return out