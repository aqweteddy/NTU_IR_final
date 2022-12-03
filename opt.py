from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--pretrained', default='google/t5-v1_1-small')
    parser.add_argument('--data', nargs='+', default=['data/train.csv'])
    parser.add_argument('--val_data', default='data/val.csv')
    parser.add_argument('--output_dir', default='ckpt')
    parser.add_argument('--template_inp', default='[q] is <s> with [r]. [q] <q> [r] <r>')
    parser.add_argument('--batch_size',type=int, default=8)
    parser.add_argument('--copy_mode', action='store_true')
    parser.add_argument('--gradient_accumulation_steps',type=int, default=4)
    
    parser.add_argument('--seed', type=int, default=1002)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--aug_prob', type=float, default=0.)
    
    return parser