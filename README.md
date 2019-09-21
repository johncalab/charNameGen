# charNameGen
This repository contains some code to generate text, which is meant to resemble human names. This is done via a simple neural network, implemented in pytorch. Essentially all of it is taken from [Rao's book](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236) on NLP, which I recommend.

To understand how the model actually works, there are a few jupyter notebooks which might be useful.
- `quickOverview.ipynb` explains how all the pieces fit together.
- `RNNlayer.ipynb` is a short theoretical explanation of how RNNs work, followed by a pytorch implementation.
- Pytorch's embedding layer can be confusing, and `embeddinglayer.ipynb` explains its basic use.
- `loadingsaving.ipynb` is a super small notebook to remind myself how to save and load models (and pickle dictionaries).

### Name Generator
The file `generate_name.py` is a simple script to generate names, using a pre-trained model. Three such models can be found in the `models` folder. The script relies on `charmodel.py` and `charvocab.py`.

For example:
- `firstnames148` was trained for 88 epochs and produces names like: Liena, Halya, Zalazi.
- `lastnames447` was trained for 88 epochs, and produces names like: Suzajiyama, Takaya, Zhingda.
- `fullnames326` was trained for only 2 epochs, and produces names like: 

I guess there are so many examples in `fullnames.txt` it overfits pretty quickly. It also learns how to capitalize automatically, which is nice.


### Source
The folder `source` contains a list a of first names, a list of last names, and a `fullnames.txt` file which is just the combination of both.
These files can be used to train models.

### Training
The file `chartrainer.py` is a script to train a model to generate names. It relies on all other .py files, and expects a choice of a source, contained in the `source` folder.

## What's missing
- **Conditioning**. Chapter 7 of Rao's book includes the option of changing the text the model spits out, dependent on a conditioning factor. For example, we might want to train names differently depending on nationality. To this end, one could proceed as follows. Training data should contain labels for each nationality. We should think of the labels as one-hot-encoded vectors. The model should then contain an extra embedding layer, spitting out a vector the same size as the hidden vector of the RNN. In turn, this vector is fed as $h_0$ of the RNN, acting as a *conditioning* factor. Something like this.
```
self.nation_embedding = nn.Embedding(embedding_dim=rnn_hidden_size, 
                                             num_embeddings=num_nationalities)
nationality_embedded = self.nation_emb(nationality_index).unsqueeze(0)
y_out, _ = self.rnn(x_embedded, nationality_embedded)
```
- All of Chapter 8 (arguably the most interesting chapter): bidirectional RNNs, attention, ...
