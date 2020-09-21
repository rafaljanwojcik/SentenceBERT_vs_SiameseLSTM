import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import sys
import logging
from tqdm.notebook import tqdm

from .models import SiameseBERT, ClassifierBERT

def setup_logger(path) -> logging.Logger:
    logger = logging.getLogger(__name__)
    setattr(logger, 'out_path', '/'.join(path.split('/')[:-1]))
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.handlers.WatchedFileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def collate_fn_lstm(batch):
	padding_list = [torch.LongTensor(item[0]) for item in batch]
	batch_size = len(padding_list)
	[padding_list.append(torch.LongTensor(item[1])) for item in batch]
	data = pad_sequence(padding_list, padding_value=0).T
	sentences1 = data[:batch_size]
	sentences2 = data[batch_size:]
	labels = [item[2] for item in batch]
	return [torch.stack([sentences1, sentences2]), torch.tensor(labels)]

def collate_fn_bert(batch):
    batch = np.array(batch)
    sent1 = batch[:, 0].tolist()
    sent2 = batch[:, 1].tolist()
    target = batch[:, 2].tolist()
    return [sent1, sent2, torch.tensor(target)]

class Trainer():
    def __init__(self, model_type: str):
        self.model_type = model_type
    def train(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, gradient_clipping_norm, epoch, logger):
        model.train()   
        for epoch_iteration, (batch) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            if isinstance(model, SiameseLSTM):
                X_batch = batch[0].to(device)
                y_batch = batch[1].float().to(device)
            elif isinstance(model, SiameseBERT) or isinstance(model, ClassifierBERT):
                pass
            else:
                raise ValueError('Incorrect model type!')
            y_hat = model(X_batch)
            preds_train.append(((y_hat>=0.5).float()==y_batch).sum().item())

            loss = criterion(y_hat, y_batch)
            loss.backward()

            epoch_loss.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

            optimizer.step()

            if epoch_iteration % 1000 == 0.:
              logger.info('Mean loss till {}th iteration of epoch {}: {}'.format(epoch_iteration, epoch, np.mean(epoch_loss)))

def train(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, gradient_clipping_norm, epoch, logger):
    model.train()   
    for epoch_iteration, (batch) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        
        X_batch = batch[0].to(device)
        y_batch = batch[1].float().to(device)
        y_hat = model(X_batch)
        preds_train.append(((y_hat>=0.5).float()==y_batch).sum().item())

        loss = criterion(y_hat, y_batch)
        loss.backward()

        epoch_loss.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

        optimizer.step()

        if epoch_iteration % 1000 == 0.:
          logger.info('Mean loss till {}th iteration of epoch {}: {}'.format(epoch_iteration, epoch, np.mean(epoch_loss)))

def eval(model, criterion, test_dataloader, device,  eval_loss, preds_test):
	model.eval()
	for batch in test_dataloader:
		with torch.no_grad():
			X_batch = batch[0].to(device)
			y_batch = batch[1].float().to(device)
			y_hat = model(X_batch)
			loss = criterion(y_hat, y_batch)
			eval_loss.append(loss.item())
			preds_test.append(((y_hat>=0.5).float()==y_batch).sum().item())
            
            
def train_bert(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, epoch, logger):
    model.train()   
    for epoch_iteration, (batch) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        
        y_batch = batch[2].to(device)
        y_hat = model(batch)
        
        if isinstance(model.module, ClassifierBERT):
            preds_train.append((torch.argmax(y_hat, dim=1) == y_batch).sum().item())
        elif isinstance(model.module, SiameseBERT):
            preds_train.append(((y_hat>=0.5).float()==y_batch).sum().item())
        else:
            raise ValueError('Invalid model instance for train_bert function!')
            

        loss = criterion(y_hat, y_batch)
        loss.backward()

        epoch_loss.append(loss.item())

        optimizer.step()

        if epoch_iteration % 1000 == 0.:
          logger.info('Mean loss till {}th iteration of epoch {}: {}'.format(epoch_iteration, epoch, np.mean(epoch_loss)))
            
def eval_bert(model, criterion, test_dataloader, device,  eval_loss, preds_test):
    model.eval()
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            y_batch = batch[2].to(device)
            y_hat = model(batch)
            loss = criterion(y_hat, y_batch)
            eval_loss.append(loss.item())
            if isinstance(model.module, ClassifierBERT):
                preds_test.append((torch.argmax(y_hat, dim=1) == y_batch).sum().item())
            elif isinstance(model.module, SiameseBERT):
                preds_test.append(((y_hat>=0.5).float()==y_batch).sum().item())
            else:
                raise ValueError('Invalid model instance for train_bert function!')
