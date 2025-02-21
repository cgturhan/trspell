import os
import time
from typing import List, Dict, Union

import numpy as np
import torch
from pytorch_pretrained_bert import BertAdam
import transformers

from .commons import DEFAULT_TRAINTEST_DATA_PATH
from .corrector import Corrector
from .seq_modeling.helpers import bert_tokenize_for_valid_examples, roberta_tokenize_for_valid_examples, initialize_tokenizer
from .seq_modeling.helpers import load_data, load_vocab_dict, save_vocab_dict
from .seq_modeling.helpers import train_validation_split, batch_iter, labelize, progressBar, batch_accuracy_func
from .seq_modeling.subwordbert import load_model, load_pretrained, load_pretrained_large, model_predictions, model_inference

""" corrector module """


class BertChecker(Corrector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pretrained_name_or_path = "bert-base-cased"
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

    def load_model(self, ckpt_path):
        print(f"initializing model")
        initialized_model = load_model(self.vocab)
        self.model = load_pretrained(initialized_model, ckpt_path, device=self.device)

    def correct_strings(self, mystrings: List[str], return_all=False) -> List[str]:
        self.is_model_ready()
        mystrings = bert_tokenize_for_valid_examples(mystrings, mystrings, self.bert_pretrained_name_or_path)[0]
        data = [(line, line) for line in mystrings]
        batch_size = 4 if self.device == "cpu" else 16
        return_strings = model_predictions(self.model, data, self.vocab, device=self.device, batch_size=batch_size, tokenizer = self.tokenizer)
        if return_all:
            return mystrings, return_strings
        else:
            return return_strings

    def correct_from_file(self, src, dest="./clean_version.txt"):
        """
        src = f"{DEFAULT_DATA_PATH}/traintest/corrupt.txt"
        """
        self.__model_status()
        x = [line.strip() for line in open(src, 'r')]
        y = self.correct_strings(x)
        print(f"saving results at: {dest}")
        opfile = open(dest, 'w')
        for line in y:
            opfile.write(line + "\n")
        opfile.close()
        return
    
    def correct_data(self, clean_file, corrupt_file, data_dir=""):
        self.is_model_ready()
        data_dir = DEFAULT_TRAINTEST_DATA_PATH if data_dir == "default" else data_dir
        batch_size = 4 if self.device == "cpu" else 16
        test_data = load_data(data_dir, clean_file, corrupt_file)

        preds = model_predictions(self.model,
                                    test_data,
                                    device=self.device,
                                    batch_size=batch_size,
                                    vocab=self.vocab,tokenizer = self.tokenizer)
        return preds
                

    def evaluate(self, clean_file, corrupt_file, data_dir="", return_preds = False, limit_token_length = False):
        self.is_model_ready()
        data_dir = DEFAULT_TRAINTEST_DATA_PATH if data_dir == "default" else data_dir
        model_name = self.pretrained_name_or_path 
        preds = []
        batch_size = 4 if self.device == "cpu" else 16
        for x, y, z in zip([data_dir], [clean_file], [corrupt_file]):
            print(x, y, z)
            test_data = load_data(x, y, z)
            if return_preds:
                preds = model_inference(self.model,
                                    test_data,
                                    topk=1,
                                    device=self.device,
                                    model_name = model_name,
                                    batch_size=batch_size,
                                    vocab_=self.vocab, limit_token_length = limit_token_length, tokenizer = self.tokenizer, return_preds= return_preds)
                
            else:
                _ = model_inference(self.model,
                                    test_data,
                                    topk=1,
                                    device=self.device,
                                    model_name = model_name,
                                    batch_size=batch_size,
                                    vocab_=self.vocab,limit_token_length=limit_token_length, tokenizer = self.tokenizer)
                
        if return_preds:
            return preds
        else:
            return

    def from_huggingface(self, pretrained_name_or_path, vocab: Union[Dict, str]):
        self.pretrained_name_or_path = pretrained_name_or_path
        if isinstance(vocab, str) and os.path.exists(vocab):
            self.vocab_path = vocab
            print(f"loading vocab from path:{self.vocab_path}")
            self.vocab = load_vocab_dict(self.vocab_path)
        elif isinstance(vocab, dict):
            self.vocab = vocab
        else:
            raise ValueError(f"unknown vocab type or unable to find path: {type(vocab)}")
        self.model = load_model(self.vocab, bert_pretrained_name_or_path=self.pretrained_name_or_path)
        self.tokenizer = initialize_tokenizer(pretrained_name_or_path)
        return

    def from_pretrained(self, pretrained_name_or_path, ckpt_path, vocab: Union[Dict, str]):
        self.pretrained_name_or_path = pretrained_name_or_path
        if isinstance(vocab, str) and os.path.exists(vocab):
            self.vocab_path = vocab
            print(f"loading vocab from path:{self.vocab_path}")
            self.vocab = load_vocab_dict(self.vocab_path)
        elif isinstance(vocab, dict):
            self.vocab = vocab
        else:
            raise ValueError(f"unknown vocab type or unable to find path: {type(vocab)}")
        self.model = load_model(self.vocab, bert_pretrained_name_or_path=self.pretrained_name_or_path)
        self.model.load_state_dict(torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location=self.device))
        self.tokenizer = initialize_tokenizer(pretrained_name_or_path)
        return

    def finetune(self,
                 clean_file,
                 corrupt_file,
                 data_dir="",
                 validation_split=0.2,
                 n_epochs=2,
                 start_epoch = 0,
                 model_name = 'bert-base',
                 batch_size = 16,
                 new_vocab_list: List = None):

        if new_vocab_list:
            raise NotImplementedError("Do not currently support modifying output vocabulary of the models "
                                      "in the finetune step; however, new vocab is accepted at training time.")

        # load data and split in train-validation
        data_dir = DEFAULT_TRAINTEST_DATA_PATH if data_dir == "default" else data_dir
        train_data = load_data(data_dir, clean_file, corrupt_file)
        
        val_clean_file = clean_file.replace("train", "val")
        val_corrupt_file = corrupt_file.replace("train", "val")
        
        valid_data = load_data(data_dir, val_clean_file, val_corrupt_file)
        #train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)
        print("len of train and test data: ", len(train_data), len(valid_data))

        # load vocab and model
        self.is_model_ready()

        # finetune
        #############################################
        # training and validation
        #############################################
       
        TRAIN_BATCH_SIZE, VALID_BATCH_SIZE = batch_size, batch_size
        GRADIENT_ACC = 4
        DEVICE = self.device
        START_EPOCH, N_EPOCHS = start_epoch, n_epochs
        CHECKPOINT_PATH = os.path.join(self.ckpt_path if self.ckpt_path else data_dir, "new_models",
                                       os.path.split(self.pretrained_name_or_path)[-1])
        if os.path.exists(CHECKPOINT_PATH) and start_epoch==0:
            num = 1
            while True:
                NEW_CHECKPOINT_PATH = CHECKPOINT_PATH + f"-{num}"
                if not os.path.exists(NEW_CHECKPOINT_PATH):
                    break
                num += 1
            CHECKPOINT_PATH = NEW_CHECKPOINT_PATH
        VOCAB_PATH = os.path.join(CHECKPOINT_PATH, "vocab.pkl")
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print(f"CHECKPOINT_PATH: {CHECKPOINT_PATH}")

        # running stats
        max_dev_acc, argmax_dev_acc = -1, -1
        patience = 100


        t_total = int(len(train_data) / TRAIN_BATCH_SIZE / GRADIENT_ACC * N_EPOCHS)
        if t_total == 0:
            t_total = 1


        # load parameters if not training from scratch
        if START_EPOCH > 1:
            progress_write_file = (
                open(os.path.join(CHECKPOINT_PATH, f"progress_retrain_from_epoch{START_EPOCH}.txt"), 'w')
            )
            self.model = load_pretrained(self.model, CHECKPOINT_PATH, optimizer=None)
            #optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=t_total)
            progress_write_file.write(f"Training model params after loading from path: {CHECKPOINT_PATH}\n")
        else:
            progress_write_file = open(os.path.join(CHECKPOINT_PATH, "progress.txt"), 'w')
            print(f"Training model params")
            progress_write_file.write(f"Training model params\n")
        progress_write_file.flush()

        is_bert = True if 'bert-base' in model_name else False
        # model to device
        model, vocab = self.model, self.vocab

        TOKENIZER = initialize_tokenizer(model_name)
        self.tokenizer = TOKENIZER

        # Create an optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=t_total)
        
        model.to(DEVICE)

        # train and eval
        for epoch_id in range(START_EPOCH, START_EPOCH + N_EPOCHS + 1):
            # check for patience
            if (epoch_id - argmax_dev_acc) > patience:
                print("patience count reached. early stopping initiated")
                print("max_dev_acc: {}, argmax_dev_acc: {}".format(max_dev_acc, argmax_dev_acc))
                break
            # print epoch
            print(f"In epoch: {epoch_id}")
            progress_write_file.write(f"In epoch: {epoch_id}\n")
            progress_write_file.flush()
            # train loss and backprop
            train_loss = 0.
            train_acc = 0.
            train_acc_count = 0.
            print("train_data size: {}".format(len(train_data)))
            progress_write_file.write("train_data size: {}\n".format(len(train_data)))
            progress_write_file.flush()
            train_data_iter = batch_iter(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            nbatches = int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE))
            optimizer.zero_grad()
            for batch_id, (batch_labels, batch_sentences) in enumerate(train_data_iter):
                
                st_time = time.time()
                # set batch data for bert
                if is_bert:
                    batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, _ = \
                        bert_tokenize_for_valid_examples(batch_labels, batch_sentences, TOKENIZER)
                else:
                    batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, _ = \
                        roberta_tokenize_for_valid_examples(batch_labels, batch_sentences, TOKENIZER)
                if len(batch_labels_) == 0:
                    print("################")
                    print("Not training the following lines due to pre-processing mismatch: \n")
                    print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
                    print("################")
                    continue
                else:
                    batch_labels, batch_sentences = batch_labels_, batch_sentences_
                batch_bert_inp = {k: v.to(DEVICE) for k, v in batch_bert_inp.items()}
                # set batch data for others
                batch_labels, batch_lengths = labelize(batch_labels, vocab)
                # batch_lengths = batch_lengths.to(device)
                batch_labels = batch_labels.to(DEVICE)
                # forward
                model.train()
                loss = model(batch_bert_inp, batch_bert_splits, targets=batch_labels)
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                if GRADIENT_ACC > 1:
                    loss = loss / GRADIENT_ACC
                loss.backward()
                # step
                if (batch_id + 1) % GRADIENT_ACC == 0 or batch_id >= nbatches - 1:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
                # compute accuracy in numpy
                if batch_id % 10000 == 0:
                    train_acc_count += 1
                    model.eval()
                    with torch.no_grad():
                        _, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels)
                    model.train()
                    batch_labels = batch_labels.cpu().detach().numpy()
                    batch_lengths = batch_lengths.cpu().detach().numpy()
                    ncorr, ntotal = batch_accuracy_func(batch_predictions, batch_labels, batch_lengths)
                    batch_acc = ncorr / ntotal
                    train_acc += batch_acc
                    # update progress
                progressBar(batch_id + 1,
                            int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE)),
                            ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", "avg_batch_acc"],
                            [time.time() - st_time, batch_loss, train_loss / (batch_id + 1), batch_acc,
                             train_acc / train_acc_count])
                if batch_id == 0 or (batch_id + 1) % 5000 == 0:
                    nb = int(np.ceil(len(train_data) / TRAIN_BATCH_SIZE))
                    progress_write_file.write(f"{batch_id + 1}/{nb}\n")
                    progress_write_file.write(
                        f"batch_time: {time.time() - st_time}, avg_batch_loss: {train_loss / (batch_id + 1)}, "
                        f"avg_batch_acc: {train_acc / train_acc_count}\n")
                    progress_write_file.flush()
            print(f"\nEpoch {epoch_id} train_loss: {train_loss / (batch_id + 1)}")

            # valid loss
            valid_loss = 0.
            valid_acc = 0.
            print("valid_data size: {}".format(len(valid_data)))
            progress_write_file.write("valid_data size: {}\n".format(len(valid_data)))
            progress_write_file.flush()
            valid_data_iter = batch_iter(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
            for batch_id, (batch_labels, batch_sentences) in enumerate(valid_data_iter):
                st_time = time.time()
                # set batch data for bert
                if is_bert:
                    batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, _ = \
                        bert_tokenize_for_valid_examples(batch_labels, batch_sentences, TOKENIZER)
                else:
                    batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, _ = \
                        roberta_tokenize_for_valid_examples(batch_labels, batch_sentences, TOKENIZER)

                if len(batch_labels_) == 0:
                    print("################")
                    print("Not validating the following lines due to pre-processing mismatch: \n")
                    print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
                    print("################")
                    continue
                else:
                    batch_labels, batch_sentences = batch_labels_, batch_sentences_
                batch_bert_inp = {k: v.to(DEVICE) for k, v in batch_bert_inp.items()}
                # set batch data for others
                batch_labels, batch_lengths = labelize(batch_labels, vocab)
                # batch_lengths = batch_lengths.to(device)
                batch_labels = batch_labels.to(DEVICE)
                # forward
                model.eval()
                with torch.no_grad():
                    batch_loss, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels)
                model.train()
                valid_loss += batch_loss
                # compute accuracy in numpy
                batch_labels = batch_labels.cpu().detach().numpy()
                batch_lengths = batch_lengths.cpu().detach().numpy()
                ncorr, ntotal = batch_accuracy_func(batch_predictions, batch_labels, batch_lengths)
                batch_acc = ncorr / ntotal
                valid_acc += batch_acc
                # update progress
                progressBar(batch_id + 1,
                            int(np.ceil(len(valid_data) / VALID_BATCH_SIZE)),
                            ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", "avg_batch_acc"],
                            [time.time() - st_time, batch_loss, valid_loss / (batch_id + 1), batch_acc,
                             valid_acc / (batch_id + 1)])
                if batch_id == 0 or (batch_id + 1) % 2000 == 0:
                    nb = int(np.ceil(len(valid_data) / VALID_BATCH_SIZE))
                    progress_write_file.write(f"{batch_id}/{nb}\n")
                    progress_write_file.write(
                        f"batch_time: {time.time() - st_time}, avg_batch_loss: {valid_loss / (batch_id + 1)}, "
                        f"avg_batch_acc: {valid_acc / (batch_id + 1)}\n")
                    progress_write_file.flush()
            print(f"\nEpoch {epoch_id} valid_loss: {valid_loss / (batch_id + 1)}")

            # save model, optimizer and test_predictions if val_acc is improved
            if valid_acc >= max_dev_acc:
                print(f"validation accuracy improved from {max_dev_acc:.4f} to {valid_acc:.4f}")
                # name = "model.pth.tar".format(epoch_id)
                # torch.save({
                #     'epoch_id': epoch_id,
                #     'max_dev_acc': max_dev_acc,
                #     'argmax_dev_acc': argmax_dev_acc,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict()},
                #     os.path.join(CHECKPOINT_PATH, name))
                name = "pytorch_model.bin"
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, name))
                print("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, name), epoch_id))
                save_vocab_dict(VOCAB_PATH, vocab)

                # re-assign
                max_dev_acc, argmax_dev_acc = valid_acc, epoch_id

        print(f"Model and logs saved at {CHECKPOINT_PATH}")
        return
