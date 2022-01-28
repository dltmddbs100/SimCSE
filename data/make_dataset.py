import numpy as np
import pandas as pd

data_path='/content/SimCSE/data/'
data_nli_path='/content/KorNLUDatasets/KorNLI/'
data_sts_path='/content/KorNLUDatasets/KorSTS/'


# Make train, valid, test format
def make_dataset(mode, path, data_type=True):
  sent1=[]
  sent2=[]
  target=[]

  if mode=='train':
    if data_type=='snli':
      data_type='snli_1.0_train.ko.tsv'
    else:
      data_type='multinli.train.ko.tsv'
    with open(path+data_type) as file:
      lines = file.readlines()

      for i, line in enumerate(lines):
        if i!=0:
          full=line.split('\t')
          sent1.append(full[0].replace('\n','').strip())
          sent2.append(full[1].replace('\n','').strip())
          target.append(full[2].replace('\n','').strip())

    data=pd.DataFrame({'sentence1':sent1, 'sentence2':sent2,'gold_label':target})
    data=data.drop_duplicates()
    data=data.sort_values('sentence1')

    entail_train=data[data['gold_label']=='entailment']
    contradict_train=data[data['gold_label']=='contradiction']

    entail_train=entail_train.sort_values('sentence1')
    contradict_train=contradict_train.sort_values('sentence1')
    contradict_train.columns=['sentence1','sentence3','gold_label']

    data=pd.merge(entail_train.drop('gold_label',axis=1),contradict_train[['sentence1','sentence3']],on='sentence1',how='inner')

    return data.reset_index(drop=True)

  if mode=='valid':

    with open(path+'sts-dev.tsv') as file:
      lines = file.readlines()

      for i, line in enumerate(lines):
        if i!=0:
          full=line.split('\t')
          sent1.append(full[5].replace('\n','').strip())
          sent2.append(full[6].replace('\n','').strip())
          target.append(full[4])

    data=pd.DataFrame({'sentence1':sent1, 'sentence2':sent2,'score':target})
    data=data.drop_duplicates()

    return data.reset_index(drop=True)

  if mode=='test':

    with open(path+'sts-test.tsv') as file:
      lines = file.readlines()

      for i, line in enumerate(lines):
        if i!=0:
          full=line.split('\t')
          sent1.append(full[5].replace('\n','').strip())
          sent2.append(full[6].replace('\n','').strip())
          target.append(full[4])

    data=pd.DataFrame({'sentence1':sent1, 'sentence2':sent2,'score':target})
    data=data.drop_duplicates()

    return data.reset_index(drop=True)


train_snli=make_dataset('train',data_nli_path,'snli')
train_mnli=make_dataset('train',data_nli_path,'mnli')

train=pd.concat([train_snli,train_mnli]).reset_index(drop=True)
valid=make_dataset('valid',data_sts_path)
test=make_dataset('test',data_sts_path)

train.to_csv(data_path+'train_nli.tsv',index=False,header=False,sep='\t')
valid.to_csv(data_path+'valid_sts.tsv',index=False,header=False,sep='\t')
test.to_csv(data_path+'test_sts.tsv',index=False,header=False,sep='\t')

train=pd.read_csv(data_path+'train_nli.tsv',header=None, sep='\t').dropna().reset_index(drop=True)
train.to_csv(data_path+'train_nli.tsv',index=False,header=False,sep='\t')