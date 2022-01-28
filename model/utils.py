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