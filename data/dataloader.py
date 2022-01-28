import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer


class ModelDataLoader(Dataset):
    def __init__(self, file_path, model_name, type):
        self.type = type

        """NLI"""
        self.anchor = []
        self.positive = []
        self.negative = []

        """STS"""
        self.label = []
        self.sentence_1 = []
        self.sentence_2 = []

        #  -------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.file_path = file_path

    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                self.data2tensor(line, type)

        if type == 'train':
            assert len(self.anchor) == len(self.positive) == len(self.negative)
        else:
            assert len(self.sentence_1) == len(self.sentence_2) == len(self.label)

    def data2tensor(self, line, type):
        split_data = line.split('\t')

        if type == 'train':
            anchor, positive, negative = split_data
            anchor = self.tokenizer.encode_plus(anchor,padding='max_length',max_length=50,truncation=True)
            positive = self.tokenizer.encode_plus(positive,padding='max_length',max_length=50,truncation=True)
            negative = self.tokenizer.encode_plus(negative,padding='max_length',max_length=50,truncation=True)

            self.anchor.append(anchor)
            self.positive.append(positive)
            self.negative.append(negative)

        else:
            sentence_1, sentence_2, label = split_data
            sentence_1 = self.tokenizer.encode_plus(sentence_1,padding='max_length',max_length=50,truncation=True)
            sentence_2 = self.tokenizer.encode_plus(sentence_2,padding='max_length',max_length=50,truncation=True)

            self.sentence_1.append(sentence_1)
            self.sentence_2.append(sentence_2)
            self.label.append(float(label.strip())/5.0)

    def __getitem__(self, index):

        if self.type == 'train':
            inputs = {'anchor': {
                'source': torch.LongTensor(self.anchor[index].input_ids),
                'valid_length': torch.tensor(self.anchor[index].attention_mask),
                'segment_ids': torch.LongTensor(self.anchor[index].token_type_ids)
                },
                      'positive': {
                'source': torch.LongTensor(self.positive[index].input_ids),
                'valid_length': torch.tensor(self.positive[index].attention_mask),
                'segment_ids': torch.LongTensor(self.positive[index].token_type_ids)
                },
                      'negative': {
                'source': torch.LongTensor(self.negative[index].input_ids),
                'valid_length': torch.tensor(self.negative[index].attention_mask),
                'segment_ids': torch.LongTensor(self.negative[index].token_type_ids)
                }}
        else:

            inputs = {'sentence_1': {
                'source': torch.LongTensor(self.sentence_1[index].input_ids),
                'valid_length': torch.tensor(self.sentence_1[index].attention_mask),
                'segment_ids': torch.LongTensor(self.sentence_1[index].token_type_ids)
                },
                      'sentence_2': {
                'source': torch.LongTensor(self.sentence_2[index].input_ids),
                'valid_length': torch.tensor(self.sentence_2[index].attention_mask),
                'segment_ids': torch.LongTensor(self.sentence_2[index].token_type_ids)
                },
                      'label': torch.FloatTensor([self.label[index]])}

        return inputs

    def __len__(self):
        if self.type == 'train':
            return len(self.anchor)
        else:
            return len(self.label)


def get_loader(args, types):

  if types == 'train':
    train_iter = ModelDataLoader(args['path_to_train_data'], args['model_name'], type='train')
    valid_iter = ModelDataLoader(args['path_to_valid_data'], args['model_name'], type='valid')

    train_iter.load_data('train')
    valid_iter.load_data('valid')

    loader = {'train': DataLoader(dataset=train_iter,
                                  batch_size=args['batch_size'],
                                  shuffle=True, pin_memory=True),
                'valid': DataLoader(dataset=valid_iter,
                                  batch_size=args['batch_size'],
                                  shuffle=True, pin_memory=True)}

  else:
    test_iter = ModelDataLoader(args['path_to_test_data'], args['test_tokenizer'], type='test')
    test_iter.load_data('test')

    loader = {'test': DataLoader(dataset=test_iter,
                                  batch_size=args['batch_size'],
                                  shuffle=False, pin_memory=True)}

  return loader