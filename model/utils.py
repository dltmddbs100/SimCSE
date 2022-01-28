from argparse import ArgumentParser


class Metric():

    def __init__(self, args):
        self.args = args

    def get_lr(self, optimizer):
        return optimizer.state_dict()['param_groups'][0]['lr']

    def count_parameters(self, model):
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def cal_dev_score(self, step, indicator):
        for key, value in indicator.items():
            indicator[key] /= step

        print("\n\nCosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            indicator['eval_pearson_cosine'], indicator['eval_spearman_cosine']))
        print("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            indicator['eval_pearson_manhattan'], indicator['eval_spearman_manhattan']))
        print("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            indicator['eval_pearson_euclidean'], indicator['eval_spearman_euclidean']))
        print("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}\n".format(
            indicator['eval_pearson_dot'], indicator['eval_spearman_dot']))

    def update_indicator(self, indicator, score):
        for key, value in indicator.items():
            if key == 'eval_spearman_cosine':
                indicator[key] += score['eval_spearman_cosine']
            elif key == 'eval_pearson_cosine':
                indicator[key] += score['eval_pearson_cosine']
            elif key == 'eval_spearman_manhattan':
                indicator[key] += score['eval_spearman_manhattan']
            elif key == 'eval_pearson_manhattan':
                indicator[key] += score['eval_pearson_manhattan']
            elif key == 'eval_spearman_euclidean':
                indicator[key] += score['eval_spearman_euclidean']
            elif key == 'eval_pearson_euclidean':
                indicator[key] += score['eval_pearson_euclidean']
            elif key == 'eval_spearman_dot':
                indicator[key] += score['eval_spearman_dot']
            elif key == 'eval_pearson_dot':
                indicator[key] += score['eval_pearson_dot']


class Argument():
  def __init__(self):
    self.parser=ArgumentParser()

  def add_args(self):
    self.parser.add_argument('--model_name', type=str, default='klue/bert-base')
    self.parser.add_argument('--weight_path', type=str, default='weights/')
    self.parser.add_argument('--path_to_train_data', type=str, default='data/train_nli.tsv')
    self.parser.add_argument('--path_to_valid_data', type=str, default='data/valid_sts.tsv')
    self.parser.add_argument('--path_to_test_data', type=str, default='data/test_sts.tsv')
    self.parser.add_argument('--device', type=str, default='cuda')
    self.parser.add_argument('--temperature', type=float, default='0.05')
    self.parser.add_argument('--batch_size', type=int, default='128')
    self.parser.add_argument('--max_epochs', type=int, default='3')
    self.parser.add_argument('--learning_rate', type=float, default='5e-5')
    self.parser.add_argument('--test_model_name', type=str, default='')
    self.parser.add_argument('--test_tokenizer', type=str, default='klue/bert-base')
    self.parser.add_argument('--train', type=str, default='True')
    self.parser.add_argument('--test', type=str, default='False')
    args = self.parser.parse_args()

    return args
