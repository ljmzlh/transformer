import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run():

    parser = argparse.ArgumentParser()

    parser.add_argument('-restore',default=None)
    parser.add_argument('-datapath', default='./dataset')
    parser.add_argument('-mode',default='train')

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-b', type=int, default=2048)
    parser.add_argument('-grad_step',type=int,default=1)
    parser.add_argument('-save_step', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=1)
    parser.add_argument('-topk',type=int,default=5)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layer', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-no_cuda', action='store_true')


    opt = parser.parse_args()

    from main import main
    main(opt)

    
if __name__ == '__main__':
    run()
