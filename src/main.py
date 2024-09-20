from Optimizor import Optimizor
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import numpy as np
import pandas as pd
import constants

import argparse
import os.path as osp
import os

def try_optimize(file_path, offset, theta = 1.0, output_dir = "./", cost_func = 'PEAK_SHIFTING_AND_HIDDEN'):
    # Input parameters:
    # offset: the overlap degree, [0,1]
    # theta : hyperparameter to balance peak_dis and hidden_dis in optimization goal
    odata = read_data(file_path)
    optimizor = Optimizor()
    data, ridgeNames, startX = optimizor.process_data(odata)
    # introduce a virtual node 0, dis[0][any] = dis[any][0] = 0
    dis = optimizor.compute_dis(data, offset, theta, cost_func)
    N = len(dis)
    nodeIDs = [x for x in range(N)]
    ans = optimizor.solve(nodeIDs, dis, False, None, None)
    rmap, seq = optimizor.deal_ans(ans, ridgeNames)
    file_name = file_path.split('/')[-1].split(".")[0]
    visualize(seq, data, offset, theta, cost_func, file_name, output_dir)


def visualize(seq, data, offset, theta, cost_func, file_path, output_dir):
    fig = plt.figure(figsize=constants.DEFAULT_FIG_SIZE)
    N,M = data.shape
    X = [x for x in range(M)]

    gs = (grid_spec.GridSpec(N,1))
    ax_objs = []
    for i in range(len(seq)):
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        ax_objs[-1].fill_between(X, data[seq[i]], color="Slateblue")
        # ax_objs[-1].set_xlim(0, N+1)
        # ax_objs[-1].set_ylim(0,2.2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(osp.join(output_dir,f'./{file_path}_offset_{offset}_theta_{theta}_func_{cost_func}.png'))

def read_data(filepath):
    df = pd.read_csv(filepath).T
    return df
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameters for update __main__")
    
    parser.add_argument('--filepath', type=str, help='File path to csv data file', default="./dataset/10_20_1.csv")
    parser.add_argument('--outputdir',type=str, help='File path to output file', default="./output")
    parser.add_argument('--costfunc' ,type=str, help='Cost Function in distance computation', default='PEAK_SHIFTING_AND_HIDDEN')
    parser.add_argument('--offset',type=float, help='Hyperparameter for the overlap degree',  default=0.5)
    parser.add_argument('--theta' ,type=float, help='Hyperparameter to balance peak_dis and hidden_dis in optimization goal', default=0.5)
    
    args = parser.parse_args()
    
    file_path, output_dir = args.filepath, args.outputdir
    offset, theta, cost_func = args.offset, args.theta, 'PEAK_SHIFTING_AND_HIDDEN'
        
    try_optimize(file_path, offset, theta, output_dir, cost_func)