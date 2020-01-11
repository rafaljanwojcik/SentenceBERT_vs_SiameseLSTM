import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def generate_batch(batch):
	padding_list = [torch.LongTensor(item[0]) for item in batch]
	batch_size = len(padding_list)
	[padding_list.append(torch.LongTensor(item[1])) for item in batch]
	data = pad_sequence(padding_list, padding_value=0).T
	sentences1 = data[:batch_size]
	sentences2 = data[batch_size:]
	labels = [item[2] for item in batch]
	return [torch.stack([sentences1, sentences2]), torch.tensor(labels)]


def train(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, gradient_clipping_norm, epoch):
	epoch_iteration = 0
	model.train()   
	for batch in train_dataloader:
		model.zero_grad()
		epoch_iteration += 1

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
		  print(f'Mean loss till {epoch_iteration}th iteration of epoch {epoch}: ', np.mean(epoch_loss))


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