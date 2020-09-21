import os
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from datetime import date
import argparse
import matplotlib.pyplot as plt

from modules.data.data import ImportData, QuoraQuestionDataset
from modules.models.embeddings import EmbeddedVocab
from modules.models.models import SiameseLSTM, ClassifierBERT
from modules.utils.utils import collate_fn_bert, train, eval, setup_logger


today = str(date.today())
path = Path(f'./logs/train_job_{today}/')
emb_path = Path('./logs/embeddings')
data_path = Path('./logs/data')
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-log", "--logdir", type=str, help="Directory to save all downloaded files, and model checkpoints.", default=path)  
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default=data_path/"dataset.csv")
    parser.add_argument("-s", "--split_seed", type=int, help="Seed for splitting the dataset.", default=44)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=2)
    parser.add_argument("-epo", "--n_epoch", type=int, help="Number of epochs.", default=4)
    parser.add_argument("-bert_cls", "--bert_class", type=str, help="Type of BERT trained (classificator, siamese).", default='siamese')
    parser.add_argument("-bert_backbone", "--bert_backbone", type=str, help="Either path to the model, or name of the BERT model that should be used, compatible with HuggingFace Transformers.", default='bert-base-uncased')

    args = parser.parse_args()
    args.logdir = args.logdir/args.model_name
    model_path = args.logdir/'best_model/'
    if not args.logdir.exists():
        os.makedirs(args.logdir)
        
    logger = setup_logger(str(args.logdir/'logs.log'))
    logger.info("Begining job. All files and logs will be saved at: {}".format(args.logdir))

        
    logger.info('Reading Dataset and splitting into train and test datasets with seed: {}'.format(args.split_seed))
    data = ImportData(str(args.data_file))
    data.train_test_split(seed=args.split_seed)


    logger.info('')
    logger.info('Number of training samples        :{}'.format(len(data.train)))
    logger.info('Number of validation samples      :{}'.format(len(data.test)))
    logger.info('')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(dataset.train.values, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn_bert)
    test_dataloader = DataLoader(dataset.test.values, batch_size= args.batch_size, shuffle=False, collate_fn=collate_fn_bert)

    model = SiameseBERT(args.bert_backbone, device) if args.bert_cls=='siamese' else ClassifierBERT(args.bert_backbone, device)
    model = model.float()
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss() if args.bert_cls=='siamese' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    logger.info('Building model.')
    logger.info('--------------------------------------')
    logger.info('Model Parameters:')
    logger.info('Hidden Size                  :{}'.format(args.n_hidden))
    logger.info('Number of layers             :{}'.format(args.n_layer))
    logger.info('Use pretrained Embeddings    :{}'.format(args.use_pretrained))
    logger.info('Dimensions of Embeddings     :{}'.format(args.emb_dim))
    logger.info('Train/fine tune Embeddings   :{}'.format(args.train_embeddings))
    logger.info('Gradient clipping            :{}'.format(args.gradient_clipping_norm))
    logger.info('--------------------------------------')
    logger.info('Training Parameters:')
    logger.info('Device                       :{}'.format(' GPU' if torch.cuda.is_available() else ' CPU'))
    logger.info('Optimizer                    :{}'.format(' Adam'))
    logger.info('Loss function                :{}'.format(' MSE'))
    logger.info('Batch Size                   :{}'.format(args.batch_size))
    logger.info('Number of Epochs             :{}'.format(args.n_epoch))
    logger.info('--------------------------------------')

    start = time()
    all_train_losses = []
    all_test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.5
    logger.info("Training the model...")
    for epoch in range(n_epoch):
        epoch_time = time()
        epoch_iteration = 0
        epoch_loss=[]
        preds_train = []

        train(model, optimizer, criterion, train_dataloader, device, epoch_loss, preds_train, args.gradient_clipping_norm, epoch, logger)

        eval_loss = []
        preds_test = []
        eval(model, criterion, test_dataloader, device, eval_loss, preds_test)

        train_loss = np.mean(epoch_loss)
        train_accuracy = np.sum(preds_train)/data.train.shape[0]
        test_loss = np.mean(eval_loss)
        test_accuracy = np.sum(preds_test)/data.test.shape[0]
        
        if test_accuracy>best_acc:
            if not model_path.exists():
                os.mkdir(model_path)
            logger.info('Saving best model at: {}'.format(str(model_path/'checkpoint.pth')))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'test_accuracy':test_accuracy
                }, str(model_path/'checkpoint.pth'))

        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        logger.info('Mean loss and accuracy of epoch {} - train: {}, {}, test: {}, {}. Calculation time: {} hours'.format(epoch, train_loss, round(train_accuracy, 4), test_loss, round(test_accuracy, 4), (time() - epoch_time)/3600))

    logger.info("Model training finished in: {}".format(np.round((time()-start)/60, 3)))

    plt.figure(figsize=(10,6))
    plt.title(f'Train and test losses during training of {args.model_name} model')
    plt.plot(list(range(len(all_train_losses))), all_train_losses, label='train')
    plt.plot(list(range(len(all_test_losses))), all_test_losses, label='test')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.logdir/'loss_plots.png')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.title(f'Train and test losses during training of {args.model_name} model')
    plt.plot(list(range(len(train_accuracies))), train_accuracies, label='train')
    plt.plot(list(range(len(test_accuracies))), test_accuracies, label='test')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(args.logdir/'acc_plots.png')
    plt.show()