import random
import numpy as np

# PyTorch
import torch

from trainer.trainer import Trainer
from utils.parser import Config, get_argparser
from utils.tasks import get_tasks

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(opts : Config):

    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    opts.target_cls = [get_tasks(opts.dataset, opts.task, step) for step in range(opts.curr_step+1)]

    opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:]


    print("==============================================")
    print(opts.num_classes)
    print(opts.target_cls)
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print(opts)
    print("==============================================")
    
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed) 

    trainer = Trainer(opts)   
    trainer.train()

if __name__ == '__main__':
    opts = get_argparser()
    main(opts)



