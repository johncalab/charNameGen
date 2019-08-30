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