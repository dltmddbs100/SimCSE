import torch
import torch.optim as optim
from torch import nn

from transformers import AutoTokenizer

from data.dataloader import ModelDataLoader, get_loader
from model.model import SimCSE
from model.loss import Loss
from model.utils import Metric
from trainer import Trainer, Tester
from SemanticSearch import semantic_search

args={}
args['model_name']='klue/bert-base'
args['weight_path']='weights/'
args['test_model_name']=''
args['test_tokenizer']=''
args['path_to_train_data']='data/train_nli.tsv'
args['path_to_valid_data']='data/valid_sts.tsv'
args['path_to_test_data']='data/test_sts.tsv'
args['device']="cuda" if torch.cuda.is_available() else "cpu"
args['temperature']=0.05
args['batch_size']=128
args['max_epochs']=3
args['learning_rate']=5e-5

# Get train dataloader
data_loader=get_loader(args,'train')

# Define model
model=SimCSE(args, mode='train').to(args['device'])

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
  {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    'weight_decay': 0.01},
  {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    'weight_decay': 0.0}
]

args['criterion']= nn.CrossEntropyLoss()
args['optimizer'] = optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'])

loss=Loss(args)
metric=Metric(args)


# Train
Trainer(args, data_loader, model, loss, metric)


# Test
args['test_model_name']='klue_bert-base_epochs:1_cosim:0.9223'
args['test_tokenizer']='klue/bert-base'

Tester(args, loss, metric)


# Semantic Search
args['test_model_name']='klue_bert-base_epochs:1_cosim:0.9223'
args['test_tokenizer']='klue/bert-base'

model=SimCSE(args, mode='test').to(args['device'])
tokenizer=AutoTokenizer.from_pretrained(args['test_tokenizer'])

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.']

# Corpus with example sentences
queries = ['한 남자가 파스타를 먹는다.',
           '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
           '치타가 들판을 가로 질러 먹이를 쫓는다.']

semantic_search(5, args, tokenizer, model, corpus, queries)