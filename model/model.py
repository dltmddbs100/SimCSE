import os

import torch
from torch import nn

from transformers import BertModel, RobertaModel


class SimCSE(nn.Module):
    def __init__(self, args, mode):
        super(SimCSE, self).__init__()
        self.args=args
        if mode == 'train':
          self.model = BertModel.from_pretrained(args.model_name)
        if mode == 'test':
          self.model = BertModel.from_pretrained(os.path.join(args.weight_path, args.test_model_name))

    def forward(self, inputs, mode):

        if mode == 'train':
            anchor_pooler = self.model(input_ids=inputs['anchor']['source'].to(self.args.device),
                                         token_type_ids=inputs['anchor']['segment_ids'].to(self.args.device),
                                         attention_mask=inputs['anchor']['valid_length'].to(self.args.device))[1]

            positive_pooler = self.model(input_ids=inputs['positive']['source'].to(self.args.device),
                                           token_type_ids=inputs['positive']['segment_ids'].to(self.args.device),
                                           attention_mask=inputs['positive']['valid_length'].to(self.args.device))[1]

            negative_pooler = self.model(input_ids=inputs['negative']['source'].to(self.args.device),
                                           token_type_ids=inputs['negative']['segment_ids'].to(self.args.device),
                                           attention_mask=inputs['negative']['valid_length'].to(self.args.device))[1]

            return anchor_pooler, positive_pooler, negative_pooler

        else:
            sentence_1_pooler = self.model(input_ids=inputs['sentence_1']['source'].to(self.args.device),
                                             token_type_ids=inputs['sentence_1']['segment_ids'].to(self.args.device),
                                             attention_mask=inputs['sentence_1']['valid_length'].to(self.args.device))[1]

            sentence_2_pooler = self.model(input_ids=inputs['sentence_2']['source'].to(self.args.device),
                                             token_type_ids=inputs['sentence_2']['segment_ids'].to(self.args.device),
                                             attention_mask=inputs['sentence_2']['valid_length'].to(self.args.device))[1]

            return sentence_1_pooler, sentence_2_pooler

    def encode(self, inputs, device):

        embeddings = self.model(input_ids=inputs['input_ids'].to(self.args.device),
                               token_type_ids=inputs['token_type_ids'].to(self.args.device),
                               attention_mask=inputs['attention_mask'].to(self.args.device))[1]

        return embeddings
