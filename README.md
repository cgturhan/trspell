<h1 align="center">
<p>Spelling correction in Turkish using toolkits and LLMs
</h1>

Paper: *Leveraging large language models for spelling correction in Turkish*  
[View the paper (PeerJ)](https://peerj.com/articles/cs-2889)  
# Contents

- [Installations](#Installation)
- [Download Datasets](#Dataset:NoisedWikiTr)
- Toolkits
    - [Neuspell](#Neuspell)
    - [ContextualSpell Checker](#ContextualSpell-Checker)
    - [Jamspell](#Jamspell) 
    - [Download Checkpoints](#Download-Checkpoints)
- LLMs
    - [Download Checkpoints](#Download-Checkpoints)
- [Tr Dictionaries](#Tr-Dictionaries)
- [Additional requirements](#Additional-requirements)

# Installation 

For Neuspell:

```bash
git clone https://github.com/neuspell/neuspell; cd neuspell
pip install -e .
```

Installation through pip:
```bash
pip install neuspell
```

For Contextual Spell Check:
``bash
pip install contextualSpellCheck
```

For Spacy Turkish model:
```bash
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl
```

For Symspell:
```bash
!pip install -U symspellpy
```

For Hunspell:
```bash
!sudo apt-get install libhunspell-dev
!python -m pip install hunspell
```
For Zemberek 
```bash
!pip install zemberek-python
```

For Jamspell 
```bash
!sudo pip install jamspell
```
For Autocorrect:
```bash
!pip install autocorrect
```

For Aspell:
```bash
!sudo apt-get install libaspell-dev
!wget https://files.pythonhosted.org/packages/53/30/d995126fe8c4800f7a9b31aa0e7e5b2896f5f84db4b7513df746b2a286da/aspell-python-py3-1.15.tar.bz2
!tar -C . -xvf aspell-python-py3-1.15.tar.bz2
%cd aspell-python-py3-1.15/
!pip install .
!pip install build
!python -m build
```

# Dataset:NoisedWikiTr

https://drive.google.com/file/d/1-42D41CJ2GpMMOdqoI4FyRP5u274n4-q/view?usp=sharing

https://drive.google.com/file/d/1y_AVzjUp1WudEPm_Y-k5U1Jp29LnD5jf/view?usp=sharing

https://drive.google.com/file/d/13GlMx7Hy40mI3NEf9B9SBJ2qCQpidLLW/view?usp=sharing

Coming Soon ... See ```data/README.md``` for more details.


# Toolkits

## Neuspell

NeuSpell is an open-source context-aware spelling correction tookit in English, featuring 10 spell checkers evaluated on real misspellings from various sources. This repository implements Neuspell in Turkish with BERTurk, Bert multilingual and XLMRoberta models.

## ContextualSpell Checker

It primarily focuses on correcting Out of Vocabulary (OOV) words and non-word errors (NWE) by utilizing the BERT model. The goal of using BERT is to incorporate context into the correction process for OOV words.

## Jamspell

JamSpell is a state-of-the-art spellchecking library that is lightweight, fast, and precise. It takes into account the surrounding words to enhance correction accuracy.

## Download Checkpoints
```
Coming Soon ...
```

# LLMs

## Download Checkpoints
```
Coming Soon ...
```

# Tr Dictionaries
Aspell:
```
!wget https://ftp.gnu.org/gnu/aspell/dict/tr/aspell-tr-0.50-0.tar.bz2
```

Hunspell VDemir:
```
!wget https://github.com/vdemir/hunspell-tr/tr_TR.aff
!wget https://github.com/vdemir/hunspell-tr/tr_TR.dic
```

Hunspell TDD:
```
!wget https://github.com/tdd-ai/hunspell-tr/tr_TR.aff
!wget https://github.com//tdd-ai/hunspell-tr/tr_TR.dic
```

# Additional requirements

Requirements for ```Neuspell``` checker:
```bash
pip install -r neuspell-extras-requirements.txt
```

Requirements for ```Aspell``` checker:

```
wget https://files.pythonhosted.org/packages/53/30/d995126fe8c4800f7a9b31aa0e7e5b2896f5f84db4b7513df746b2a286da/aspell-python-py3-1.15.tar.bz2
tar -C . -xvf aspell-python-py3-1.15.tar.bz2
cd aspell-python-py3-1.15
python setup.py install
```

Requirements for ```Jamspell``` checker:

```
sudo apt-get install -y swig3.0
wget -P ./ https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz
tar xf ./en.tar.gz --directory ./
```

## Citation

If you use this work, please cite the following paper:

```bibtex
@article{turhan2024leveraging,
  title={Leveraging large language models for spelling correction in Turkish},
  author={G{\"u}zel Turhan, Ceren},
  journal={PeerJ Computer Science},
  volume={11},
  pages={e2889},
  year={2025},
  publisher={PeerJ}
}
