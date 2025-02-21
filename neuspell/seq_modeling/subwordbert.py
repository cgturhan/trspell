import time

from .downloads import download_pretrained_model
from .evals import get_metrics
from .helpers import *
from .models import SubwordBert
import pandas as pd


def load_model(vocab, bert_pretrained_name_or_path=None, verbose=False):
    model = SubwordBert(vocab["token2idx"][vocab["pad_token"]],
                        len(vocab["token_freq"]),
                        bert_pretrained_name_or_path=bert_pretrained_name_or_path)

    if verbose:
        print(model)
    print(f"Number of parameters in the model: {get_model_nparams(model)}")

    return model


def load_pretrained(model, checkpoint_path, optimizer=None, device='cuda'):
    if optimizer:
        raise Exception("If you want optimizer, call `load_pretrained_large(...)` instead of `load_pretrained(...)`")

    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print(f"Loading model params from checkpoint dir: {checkpoint_path}")

    try:
        checkpoint_data = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=map_location)
    except FileNotFoundError:
        download_pretrained_model(checkpoint_path)
        checkpoint_data = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=map_location)

    model.load_state_dict(checkpoint_data)

    return model


def load_pretrained_large(model, checkpoint_path, optimizer=None, device='cuda'):
    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print(f"Loading model params from checkpoint dir: {checkpoint_path}")
    checkpoint_data = torch.load(os.path.join(checkpoint_path, "model.pth.tar"), map_location=map_location)
    # print(f"previously model saved at : {checkpoint_data['epoch_id']}")

    model.load_state_dict(checkpoint_data['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    max_dev_acc, argmax_dev_acc = checkpoint_data["max_dev_acc"], checkpoint_data["argmax_dev_acc"]

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc

    return model


def model_predictions(model, data, vocab, device, tokenizer, batch_size=16):
    """
    model: an instance of SubwordBert
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    """

    topk = 1
    # print("###############################################")
    inference_st_time = time.time()
    final_sentences = []
    VALID_batch_size = batch_size
    # print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_batch_size, shuffle=False)
    model.eval()
    model.to(device)
    for batch_id, (batch_labels, batch_sentences) in enumerate(data_iter):
        # set batch data for bert
        batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, valid_idxs = bert_tokenize_for_valid_examples(
            batch_labels, batch_sentences, tokenizer)
        if len(batch_labels_) == 0:
            print("################")
            print("Not predicting the following lines due to pre-processing mismatch: \n")
            print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
            print("################")
            continue
        else:
            batch_labels, batch_sentences = batch_labels_, batch_sentences_
        batch_bert_inp = {k: v.to(device) for k, v in batch_bert_inp.items()}
        # set batch data for others
        batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
        # batch_lengths = batch_lengths.to(device)
        batch_labels_ids = batch_labels_ids.to(device)
        # forward
        with torch.no_grad():
            """
            NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
            """
            _, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids, topk=topk)
        batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_labels)
        final_sentences.extend([[sentence, prediction] for sentence, prediction in zip(batch_sentences, batch_predictions)])
    # print("total inference time for this data is: {:4f} secs".format(time.time()-inference_st_time))
    return final_sentences


def model_inference(model, data, topk, device, tokenizer, model_name = 'bert-base-cased', batch_size=16, chunk_size=72, limit_token_length = False, vocab_=None, return_preds = False):
    """
    model: an instance of SubwordBert
    data: list of tuples, with each tuple consisting of correct and incorrect 
            sentence string (would be split at whitespaces)
    topk: how many of the topk softmax predictions are considered for metrics calculations
    """
    if vocab_ is not None:
        vocab = vocab_

    topk = 5 if limit_token_length else topk
    print("###############################################")
    inference_st_time = time.time()
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    VALID_batch_size = batch_size
    valid_loss = 0.
    valid_acc = 0.
    if return_preds: 
        final_sentences = []
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_batch_size, shuffle=False)
    model.eval()
    model.to(device)
    total_batch = len(data)// VALID_batch_size
    for batch_id, (batch_labels, batch_sentences) in tqdm(enumerate(data_iter), total=total_batch, desc="Processing Batches"):
        torch.cuda.empty_cache()
        st_time = time.time()
        # set batch data for bert
        if 'bert-base' in model_name: 
            batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, valid_idx = bert_tokenize_for_valid_examples(batch_labels, batch_sentences, tokenizer)
        else:
            batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits, valid_idx = roberta_tokenize_for_valid_examples(batch_labels, batch_sentences, tokenizer)
        if len(batch_labels_) == 0:
            print("################")
            print("Not predicting the following lines due to pre-processing mismatch: \n")
            print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
            print("################")
            continue
        else:
            org_batch_sentences, org_batch_labels = batch_sentences, batch_labels
            batch_labels, batch_sentences = batch_labels_, batch_sentences_
        batch_bert_inp = {k: v.to(device) for k, v in batch_bert_inp.items()}
        # set batch data for others
        batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
        # batch_lengths = batch_lengths.to(device)
        batch_labels_ids = batch_labels_ids.to(device)
        # forward
        try:
            with torch.no_grad():
                """
                NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
                """
                batch_loss, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids,
                                                      topk=topk)
        except RuntimeError:
            print(f"batch_bert_inp:{len(batch_bert_inp.keys())},batch_labels_ids:{batch_labels_ids.shape}")
            raise Exception("")
        #print(batch_predictions.shape)
        valid_loss += batch_loss
        # compute accuracy in numpy
        batch_labels_ids = batch_labels_ids.cpu().detach().numpy()
        batch_lengths = batch_lengths.cpu().detach().numpy()
        # based on topk, obtain either strings of batch_predictions or list of tokens
        if topk == 1 and not limit_token_length:
            batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_sentences)
        elif limit_token_length:
            batch_predictions = untokenize_without_unks_(batch_predictions, batch_lengths, vocab, batch_sentences)
        else:
            batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_sentences,topk=None)

        # corr2corr, corr2incorr, incorr2corr, incorr2incorr, mistakes = \
        #    get_metrics(batch_labels,batch_sentences,batch_predictions,check_until_topk=topk,return_mistakes=True)
        # _mistakes.extend(mistakes)
        # batch_labels = [line.lower() for line in batch_labels]
        # batch_sentences = [line.lower() for line in batch_sentences]
        # batch_predictions = [line.lower() for line in batch_predictions]

        if len(valid_idx) != len(org_batch_labels):
            total_batch_sentences = [None] * len(org_batch_sentences)
            total_batch_predictions = [None] * len(org_batch_sentences)
            total_batch_labels = [None] * len(org_batch_labels)
                                   
            for val_idx, sent, preds, labels in zip(valid_idx, batch_sentences, batch_predictions, batch_labels):
                total_batch_sentences[val_idx] = sent
                total_batch_predictions[val_idx] = preds
                total_batch_labels[val_idx] = labels
            
            not_valid_idx = list(set(range(len(org_batch_labels))) - set(valid_idx))
            for idx in not_valid_idx:
                org_chunks = split_text_into_word_chunks(org_batch_labels[idx], chunk_size)
                noisy_chunks = split_text_into_word_chunks(org_batch_sentences[idx], chunk_size)
                if 'bert-base' in model_name: 
                    chunk_labels_, chunk_sentences_, chunk_bert_inp, chunk_bert_splits, chunk_valid_idx= bert_tokenize_for_valid_examples(org_chunks, noisy_chunks, tokenizer)
                else:
                    chunk_labels_, chunk_sentences_, chunk_bert_inp, chunk_bert_splits, chunk_valid_idx= roberta_tokenize_for_valid_examples(org_chunks, noisy_chunks, tokenizer)
                chunk_labels, chunk_sentences = chunk_labels_, chunk_sentences_
                chunk_bert_inp = {k: v.to(device) for k, v in chunk_bert_inp.items()}

                chunk_labels_ids, chunk_lengths = labelize(chunk_labels, vocab)
                chunk_labels_ids = chunk_labels_ids.to(device)
                # forward
                try:
                    with torch.no_grad():
                        chunk_loss, chunk_predictions = model(chunk_bert_inp, chunk_bert_splits, targets=chunk_labels_ids,topk=topk)
                except RuntimeError:
                    print(f"chunk_bert_inp:{len(chunk_bert_inp.keys())},batch_labels_ids:{chunk_labels_ids.shape}")
                    
                    raise Exception("")
                valid_loss += chunk_loss
                # compute accuracy in numpy
                chunk_labels_ids = chunk_labels_ids.cpu().detach().numpy()
                chunk_lengths = chunk_lengths.cpu().detach().numpy()

                # based on topk, obtain either strings of batch_predictions or list of tokens
                if topk == 1 and not limit_token_length:
                    chunk_predictions = untokenize_without_unks(chunk_predictions, chunk_lengths, vocab, chunk_sentences)
                elif limit_token_length:
                    chunk_predictions = untokenize_without_unks_(chunk_predictions, chunk_lengths, vocab, chunk_sentences)
                else:
                    chunk_predictions = untokenize_without_unks2(chunk_predictions, chunk_lengths, vocab, chunk_sentences,topk=None)

                combined_chunk_predictions, combined_chunk_sentences, combined_chunk_labels =  [], [], []

                for preds, sents, labels in zip(chunk_predictions, chunk_sentences, chunk_labels):
                    combined_chunk_predictions.append(preds)
                    combined_chunk_sentences.append(sents)
                    combined_chunk_labels.append(labels)
             
                assert len(combined_chunk_predictions)==len(combined_chunk_sentences) == len(combined_chunk_labels)
                combined_chunk_predictions = " ".join(combined_chunk_predictions)
                combined_chunk_sentences = " ".join(combined_chunk_sentences)
                combined_chunk_labels = " ".join(combined_chunk_labels)

                total_batch_sentences[idx] = combined_chunk_sentences
                total_batch_predictions[idx]= combined_chunk_predictions
                total_batch_labels[idx] = combined_chunk_labels

            if total_batch_sentences == org_batch_sentences and total_batch_labels == org_batch_labels :
                batch_labels, batch_sentences, batch_predictions = total_batch_labels, total_batch_sentences, total_batch_predictions        

        corr2corr, corr2incorr, incorr2corr, incorr2incorr = \
            get_metrics(batch_labels, batch_sentences, batch_predictions, check_until_topk=topk, return_mistakes=False)
        _corr2corr += corr2corr
        _corr2incorr += corr2incorr
        _incorr2corr += incorr2corr
        _incorr2incorr += incorr2incorr

        if return_preds:         
             final_sentences.extend([[sentence, prediction] for sentence, prediction in zip(batch_sentences, batch_predictions)])

        # delete
        del batch_loss
        del batch_predictions
        del batch_labels, batch_lengths, batch_bert_inp
        torch.cuda.empty_cache()

        '''
        # update progress
        progressBar(batch_id+1,
                    int(np.ceil(len(data) / VALID_batch_size)), 
                    ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
                    [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        '''
  
    
    print(f"\nEpoch {None} valid_loss: {valid_loss / (batch_id + 1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time() - inference_st_time))
    print("###############################################")
    print("")
    # for mistake in _mistakes:
    #    print(mistake)
    print("")
    print("total token count: {}".format(_corr2corr + _corr2incorr + _incorr2corr + _incorr2incorr))
    print(
        f"_corr2corr:{_corr2corr}, _corr2incorr:{_corr2incorr}, _incorr2corr:{_incorr2corr}, _incorr2incorr:{_incorr2incorr}")
    print(f"accuracy is {(_corr2corr + _incorr2corr) / (_corr2corr + _corr2incorr + _incorr2corr + _incorr2incorr)}")
    print(f"word correction rate is {(_incorr2corr) / (_incorr2corr + _incorr2incorr)}")
    print("###############################################")
    if return_preds:
        return final_sentences
    else:
        return
    
