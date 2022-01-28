import numpy as np
import time
import os

import torch
from torch import nn
from torch.cuda import amp

from data.dataloader import get_loader
from model.model import SimCSE

def Trainer(args, data_loader, model, loss, metric):

  scaler = amp.GradScaler()
  optimizer = args['optimizer']
  criterion = args['criterion']
  
  for epoch_i in range(0, args['max_epochs']):
    
    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args['max_epochs']))
    model.train()
    t0 = time.time()
    total_train_loss = 0
    total_batch=len(data_loader['train'])

    for i, batch in enumerate(data_loader['train']):

      inputs = batch

      optimizer.zero_grad()

      with amp.autocast():
        anchor_embeddings, positive_embeddings, negative_embeddings = model(inputs, 'train')
        train_loss = loss.train_loss_fct(criterion, anchor_embeddings, positive_embeddings, negative_embeddings)

      total_train_loss += train_loss.item()

      scaler.scale(train_loss).backward()
      scaler.step(optimizer)
      scaler.update()

      training_time = time.time() - t0

      print(f"\rTotal Batch {i+1}/{total_batch} , elapsed time : {training_time/(i+1):.1f}s , train_loss : {total_train_loss/(i+1):.2f}", end='')
    print("")

    # ========================================
    #               Validating
    # ========================================
    model.eval()
    total_eval_loss=0
    total_val_batch=len(data_loader['valid'])

    score_indicator = {'eval_pearson_cosine': 0,
                      'eval_spearman_cosine': 0,
                      'eval_pearson_manhattan': 0,
                      'eval_spearman_manhattan': 0,
                      'eval_pearson_euclidean': 0,
                      'eval_spearman_euclidean': 0,
                      'eval_pearson_dot': 0,
                      'eval_spearman_dot': 0}

    for i, batch in enumerate(data_loader['valid']):
      inputs = batch

      with torch.no_grad():
        sentence_1_embeddings, sentence_2_embeddings = model(inputs, 'valid')
        score = loss.evaluation(sentence_1_embeddings, sentence_2_embeddings, inputs['label'])

      metric.update_indicator(score_indicator, score)

      print(f"\rValidation Batch {i+1}/{total_val_batch}", end='')
    score=metric.cal_dev_score(i, score_indicator)
    
    model.model.save_pretrained(os.path.join(args['weight_path'], f"{args['model_name'].replace('/','_')}_epochs:{epoch_i+1}_cosim:{round(score_indicator['eval_spearman_cosine'],4)}"))


def Tester(args, loss, metric):
  data_loader=get_loader(args,'test')

  model=SimCSE(args, mode='test').to(args['device'])
  model.eval()

  total_test_loss=0
  total_test_batch=len(data_loader['test'])

  score_indicator = {'eval_pearson_cosine': 0,
                     'eval_spearman_cosine': 0,
                     'eval_pearson_manhattan': 0,
                     'eval_spearman_manhattan': 0,
                     'eval_pearson_euclidean': 0,
                     'eval_spearman_euclidean': 0,
                     'eval_pearson_dot': 0,
                     'eval_spearman_dot': 0}

  for i, batch in enumerate(data_loader['test']):
    inputs = batch

    with torch.no_grad():
      sentence_1_embeddings, sentence_2_embeddings = model(inputs, 'test')
      score = loss.evaluation(sentence_1_embeddings, sentence_2_embeddings, inputs['label'])

    metric.update_indicator(score_indicator, score)

    print(f"\rTest Batch {i+1}/{total_test_batch}", end='')
  score=metric.cal_dev_score(i, score_indicator)