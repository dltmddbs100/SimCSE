from data.dataloader import ModelDataLoader, get_loader
from model.model import SimCSE
from model.loss import Loss
from model.utils import Metric, Argument
from trainer import Trainer, Tester

def main(args):
  
  loss = Loss(args)
  metric = Metric(args)
  
  if args.train == 'True':
    # Get train dataloader
    data_loader = get_loader(args,'train')

    print('\nModel Loading...')
    model = SimCSE(args, mode='train').to(args.device)

    Trainer(args, data_loader, model, loss, metric)
    
  if args.test == 'True':
    # Get test dataloader
    data_loader = get_loader(args,'test')
    model = SimCSE(args, mode='test').to(args.device)
    
    Tester(args, data_loader, model, loss, metric)
    
    
if __name__ == '__main__':
    args = Argument().add_args()
    main(args)
