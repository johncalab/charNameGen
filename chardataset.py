"""
Assumes:
    vectorizer is a charVectorizer
    corpus is a pandas Series
        use: corpus = pd.read_csv('lastnames_clean.txt',header=None)[0]
"""
from torch.utils.data import Dataset
class charDataset(Dataset):
    def __init__(self,vectorizer,corpus):
        self.corpus = corpus
        self.vectorizer = vectorizer

        self.max_len = int(corpus.str.len().max()) + 2

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self,i):
        name = self.corpus[i]
        return self.vectorizer.vectorize(name=name, max_len=self.max_len)
