# SimCSE
- This repository contains the codes for paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821).
- Supervised SimCSE Implementation With Korean Using Pytorch

![image](https://user-images.githubusercontent.com/55730591/163663939-447829db-20f5-4ff9-9f6d-a0d27c92a6dc.png)

## Training
+ Huggingface Transformers
 
  + **You can use various pre-trained models in huggingface model hub**

+ Datasets - [kakaobrain/KorNLUDatasets](https://github.com/kakaobrain/KorNLUDatasets)
  + Train: KorNLI
    - multinli.train.ko.tsv
    - snli_1.0_train.ko.tsv
  
  + Dev/Test: KorSTS
    - sts-dev.tsv
    - sts-test.tsv

+ Setup
  + model : klue/bert-base
  + max_epochs : 3
  + temperature : 0.05
  + batch_size : 128
  + max_epochs : 3
  + learning_rate : 5e-05 

## Performance
| Model                  | AVG | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| KoSBERT<sup>†</sup><sub>SKT</sub>    | 77.40 | 78.81 | 78.47 | 77.68 | 77.78 | 77.71 | 77.83 | 75.75 | 75.22 |
| KoSBERT<sub>base</sub>               | 80.39 | 82.13 | 82.25 | 80.67 | 80.75 | 80.69 | 80.78 | 77.96 | 77.90 |
| KoSRoBERTa<sub>base</sub>            | 81.64 | 81.20 | 82.20 | 81.79 | 82.34 | 81.59 | 82.20 | 80.62 | 81.25 |
| | | | | | | | | |
| KoSimCSE-BERT<sup>†</sup><sub>SKT</sub>   | 81.32 | 82.12 | 82.56 | 81.84 | 81.63 | 81.99 | 81.74 | 79.55 | 79.19 |
| KoSimCSE-BERT<sub>base</sub>              | 81.56 | 83.05 | 83.33 | 82.62 | 82.96 | 82.78 | 83.09 | 77.97 | 76.70 |
| KoSimCSE-RoBERTa<sub>base</sub>           | 83.35 | 83.91 | 84.22 | 83.60 | 84.07 | 83.64 | 84.04 | 82.01 | 81.32 |
| SimCSE(ours)-BERT<sub>base</sub>          | **86.15** | **87.46** | **86.94** | **87.98** | **86.77** | **87.60** | **86.70** | **83.58** | **82.15** |
  
## Installation
```
!git clone https://github.com/dltmddbs100/SimCSE.git
!git clone https://github.com/kakaobrain/KorNLUDatasets.git
!pip install transformers
cd /content/SimCSE/
```

## Getting Started
**Make dataset to 'data' directory from KorNLUDatasets.** <br/>
Both multinli.train.ko.tsv and snli_1.0_train.ko.tsv are concatenated to single dataset.
```python
# 1. Make datasets from KorNLU
!python data/make_dataset.py
```
<br/>

**Train SimCSE with pre-trained model using BERT [CLS] token representation.** <br/>
Validation score tables are supported and models are saved at the end of each epoch.
```python
# 2. Train model 
!python main.py --train 'True' 
                --model_name 'klue/bert-base'
                --weight_path '' # you can assign your own path
                --path_to_train_data data/train_nli.tsv
                --path_to_valid_data data/valid_sts.tsv
                --path_to_test_data data/test_sts.tsv
                --device 'cuda'
                --temperature 0.05
                --batch_size 128
                --max_epochs 3
                --learning_rate 5e-05
                --test 'False'
```
+ `--train`: If you want to train the model, it should be 'True' while test argument is 'False'.
+ `--model_name`: The name or path of a transformers-based pre-trained checkpoint (default: klue/bert-base)
+ `--weight_path`: The place where your trained weights are saved.
+ `--device`: Supports 'cuda' or 'cpu'.
+ `--temperature`: Scaling magnitude of cosine similarity term.

<br/>

**Check the test set score with saved models.**
```python
# 3. Inference 
!python main.py --test 'True' 
                --weight_path '' # you can assign your own path
                --test_tokenizer 'klue/bert-base' 
                --test_model_name 'klue_bert-base_epochs:1_cosim:0.928'
```
+ `--test`: If you want to train the model, it should be 'True' while train argument is 'False'.
+ `--weight_path`: The place where your trained weights are saved.
+ `--test_tokenizer`: The name or path of a transformers-based pre-trained tokenizer checkpoint.
+ `--test_model_name`: The path of the model you want to make inference.


## Semantic Search
Based on trained model, semantic similarity is obtained by specifying several sentences.

```python
from SemanticSearch import semantic_search
from model.model import SimCSE

from easydict import EasyDict as edict
from transformers import AutoTokenizer

# Semantic Search
args=edict()
args['device']='cuda'
args['weight_path']= ''  # you can assign your own path
args['test_model_name']= 'klue_bert-base_epochs:1_cosim:0.928'

model = SimCSE(args, mode='test').to(args.device)
tok = AutoTokenizer.from_pretrained('klue/bert-base')

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

queries = ['한 남자가 파스타를 먹는다.',
           '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
           '치타가 들판을 가로 질러 먹이를 쫓는다.']

semantic_search(5, args, tok, model, corpus, queries)
```

**Results**
```
======================

Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:

한 남자가 빵 한 조각을 먹는다. (Score: 0.5000)
한 남자가 음식을 먹는다. (Score: 0.4874)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.1812)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.0264)
한 여자가 바이올린을 연주한다. (Score: 0.0260)

======================

Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:

원숭이 한 마리가 드럼을 연주한다. (Score: 0.5549)
한 남자가 말을 탄다. (Score: 0.2493)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.2134)
한 여자가 바이올린을 연주한다. (Score: 0.2047)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.2007)

======================

Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:

치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.6544)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.2700)
한 남자가 말을 탄다. (Score: 0.1711)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1625)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1379)
```

## References
```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```

```bibtex
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```
