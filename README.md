<h1 align="center">
<p>Spelling correction in Turkish using toolkits and LLMs
</h1>

# Contents

- [Installations](#Installation)
- [Download Datasets](#Datasets)
- Toolkits
    - [Download Checkpoints](#Download-Checkpoints)
- LLMs
    - [Download Checkpoints](#Download-Checkpoints)
- [Additional requirements](#Additional-requirements)

# Installation 

## Neuspell

```bash
git clone https://github.com/neuspell/neuspell; cd neuspell
pip install -e .
```

To install extra requirements,

```bash
pip install -r extras-requirements.txt
```

or individually as:

```bash
pip install -e .[elmo]
pip install -e .[spacy]
```

NOTE: For _zsh_, use ".[elmo]" and ".[spacy]" instead

Additionally, ```spacy models``` can be downloaded as:

```bash
python -m spacy download en_core_web_sm
```

Then, download pretrained models of `neuspell` following [Download Checkpoints](#Download-Checkpoints)

Here is a quick-start code snippet (command line usage) to use a checker model.
See [test_neuspell_correctors.py](./tests/test_neuspell_correctors.py) for more usage patterns.

```python
import neuspell
from neuspell import available_checkers, BertChecker

""" see available checkers """
print(f"available checkers: {neuspell.available_checkers()}")
# → available checkers: ['BertsclstmChecker', 'CnnlstmChecker', 'NestedlstmChecker', 'SclstmChecker', 'SclstmbertChecker', 'BertChecker', 'SclstmelmoChecker', 'ElmosclstmChecker']

""" select spell checkers & load """
checker = BertChecker()
checker.from_pretrained()

""" spell correction """
checker.correct("I luk foward to receving your reply")
# → "I look forward to receiving your reply"
checker.correct_strings(["I luk foward to receving your reply", ])
# → ["I look forward to receiving your reply"]
checker.correct_from_file(src="noisy_texts.txt")
# → "Found 450 mistakes in 322 lines, total_lines=350"

""" evaluation of models """
checker.evaluate(clean_file="bea60k.txt", corrupt_file="bea60k.noise.txt")
# → data size: 63044
# → total inference time for this data is: 998.13 secs
# → total token count: 1032061
# → confusion table: corr2corr:940937, corr2incorr:21060,
#                    incorr2corr:55889, incorr2incorr:14175
# → accuracy is 96.58%
# → word correction rate is 79.76%
```

Alternatively, once can also select and load a spell checker differently as follows:

```python
from neuspell import SclstmChecker

checker = SclstmChecker()
checker = checker.add_("elmo", at="input")  # "elmo" or "bert", "input" or "output"
checker.from_pretrained()
```

This feature of adding ELMO or BERT model is currently supported for selected models.
See [List of neural models in the toolkit](#List-of-neural-models-in-the-toolkit) for details.

If interested, follow [Additional Requirements](#Additional-requirements) for installing non-neural spell
checkers- ```Aspell``` and ```Jamspell```.

### Installation through pip

```bash
pip install neuspell
```
## Symspell
### Installation through pip
!pip install -U symspellpy

## Hunspell
### Installation through pip
!sudo apt-get install libhunspell-dev
!python -m pip install hunspell

## Zemberek 
### Installation through pip
!pip install zemberek-python

## Jamspell 
### Installation through pip
!sudo pip install jamspell


### Datasets

We curate several synthetic and natural datasets for training/evaluating neuspell models. For full details, check
our [paper](#Citation). Run the following to download all the datasets.

```
cd data/traintest
python download_datafiles.py 
```

See ```data/traintest/README.md``` for more details.

Train files are dubbed with names ```.random```, ```.word```, ```.prob```, ```.probword``` for different noising
startegies used to create them. For each strategy (see [Synthetic data creation](#Synthetic-data-creation)), we noise
∼20% of the tokens in the clean corpus. We use 1.6 Million sentences from
the [```One billion word benchmark```](https://arxiv.org/abs/1312.3005) dataset as our clean corpus.



example_texts = [
    "This is an example sentence to demonstrate noising in the neuspell repository.",
    "Here is another such amazing example !!"
]

word_repl_noiser = WordReplacementNoiser(language="english")
word_repl_noiser.load_resources()
noise_texts = word_repl_noiser.noise(example_texts)
print(noise_texts)
```

##### Other languages

```
Coming Soon ...
```

# Finetuning on custom data and creating new models

### Finetuning on top of `neuspell` pretrained models

```python
from neuspell import BertChecker

checker = BertChecker()
checker.from_pretrained()
checker.finetune(clean_file="sample_clean.txt", corrupt_file="sample_corrupt.txt", data_dir="default")
```

This feature is only available for `BertChecker` and `ElmosclstmChecker`.

### Training other Transformers/BERT-based models

We now support initializing a huggingface model and finetuning it on your custom data. Here is a code snippet
demonstrating that:

First mark your files containing clean and corrupt texts in a line-seperated format

```python
from neuspell.commons import DEFAULT_TRAINTEST_DATA_PATH

data_dir = DEFAULT_TRAINTEST_DATA_PATH
clean_file = "sample_clean.txt"
corrupt_file = "sample_corrupt.txt"
```

```python
from neuspell.seq_modeling.helpers import load_data, train_validation_split
from neuspell.seq_modeling.helpers import get_tokens
from neuspell import BertChecker

# Step-0: Load your train and test files, create a validation split
train_data = load_data(data_dir, clean_file, corrupt_file)
train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)

# Step-1: Create vocab file. This serves as the target vocab file and we use the defined model's default huggingface
# tokenizer to tokenize inputs appropriately.
vocab = get_tokens([i[0] for i in train_data], keep_simple=True, min_max_freq=(1, float("inf")), topk=100000)

# # Step-2: Initialize a model
checker = BertChecker(device="cuda")
checker.from_huggingface(bert_pretrained_name_or_path="distilbert-base-cased", vocab=vocab)

# Step-3: Finetune the model on your dataset
checker.finetune(clean_file=clean_file, corrupt_file=corrupt_file, data_dir=data_dir)
```

You can further evaluate your model on a custom data as follows:

```python
from neuspell import BertChecker

checker = BertChecker()
checker.from_pretrained(
    bert_pretrained_name_or_path="distilbert-base-cased",
    ckpt_path=f"{data_dir}/new_models/distilbert-base-cased"  # "<folder where the model is saved>"
)
checker.evaluate(clean_file=clean_file, corrupt_file=corrupt_file, data_dir=data_dir)
```

### Multilingual Models

Following usage above, once can now seamlessly utilize multilingual models such as `xlm-roberta-base`,
`bert-base-multilingual-cased` and `distilbert-base-multilingual-cased` on a non-English script.


# Toolkits

### Neuspell

NeuSpell is an open-source context-aware spelling correction tookit in English, featuring 10 spell checkers evaluated on real misspellings from various sources. This repository implements Neuspell in Turkish with BERTurk, Bert multilingual and XLMRoberta models.

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

# Citation

```
@inproceedings{jayanthi-etal-2020-neuspell,
    title = "{N}eu{S}pell: A Neural Spelling Correction Toolkit",
    author = "Jayanthi, Sai Muralidhar  and
      Pruthi, Danish  and
      Neubig, Graham",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.21",
    doi = "10.18653/v1/2020.emnlp-demos.21",
    pages = "158--164",
    abstract = "We introduce NeuSpell, an open-source toolkit for spelling correction in English. Our toolkit comprises ten different models, and benchmarks them on naturally occurring misspellings from multiple sources. We find that many systems do not adequately leverage the context around the misspelt token. To remedy this, (i) we train neural models using spelling errors in context, synthetically constructed by reverse engineering isolated misspellings; and (ii) use richer representations of the context. By training on our synthetic examples, correction rates improve by 9{\%} (absolute) compared to the case when models are trained on randomly sampled character perturbations. Using richer contextual representations boosts the correction rate by another 3{\%}. Our toolkit enables practitioners to use our proposed and existing spelling correction systems, both via a simple unified command line, as well as a web interface. Among many potential applications, we demonstrate the utility of our spell-checkers in combating adversarial misspellings. The toolkit can be accessed at neuspell.github.io.",
}
```

[Link](https://www.aclweb.org/anthology/2020.emnlp-demos.21/) for the publication. Any questions or suggestions, please
contact the authors at jsaimurali001 [at] gmail [dot] com
