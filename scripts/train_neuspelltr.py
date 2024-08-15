from neuspell.commons import DEFAULT_TRAINTEST_DATA_PATH
from neuspell.seq_modeling.helpers import load_data, train_validation_split
from neuspell.seq_modeling.helpers import get_tokens
from neuspell import BertChecker
from transformers import AutoTokenizer


data_dir = DEFAULT_TRAINTEST_DATA_PATH
clean_file = "wiki_tr_train_clean.txt"
corrupt_file = "wiki_tr_train_corrupt.txt"
model_name = "dbmdz/bert-base-turkish-128k-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step-0: Load your train and test files, create a validation split
train_data = load_data(data_dir, clean_file, corrupt_file)
#train_data, _ = train_validation_split(train_data, 1.0, seed=11690)

# Step-1: Create vocab file. This serves as the target vocab file and we use the defined model's default huggingface
# tokenizer to tokenize inputs appropriately.
vocab = get_tokens([i[0] for i in train_data], keep_simple=True, min_max_freq=(1, float("inf")), topk=100000)

# # Step-2: Initialize a model
checker = BertChecker(device="cuda")
checker.from_huggingface(bert_pretrained_name_or_path=model_name, vocab=vocab)

# Step-3: Finetune the model on your dataset
checker.finetune(clean_file=clean_file, corrupt_file=corrupt_file, data_dir=data_dir)