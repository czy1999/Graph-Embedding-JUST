import numpy as np
import networkx as nx
import random
import math
import time
import argparse
from tqdm import tqdm
from gensim.models import Word2Vec


def parse_args():
    parser = argparse.ArgumentParser(description="Ju&st")

    parser.add_argument('--input' , dest = 'input_file', default = 'dblp.edgelist')

    parser.add_argument('--output', dest = 'output_file',default = 'data.embeddings')

    parser.add_argument('--dimensions' ,dest = 'dimensions', type=int, default = 128)

    parser.add_argument('--walk_length',  dest = 'walk_length', type=int, default = 10)

    parser.add_argument('--num_walks' ,dest = 'num_walks', type=int, default = 1)

    parser.add_argument('--window_size' , dest = 'window_size',type=int, default = 10)
    
    parser.add_argument('--alpha' , type=float, default = 0.5)

    return parser.parse_args()


def get_justwalk(G,node,L_max,a,m=2):
    walk = [node]
    Q_hist = [node[0]]
    Pr_stay = 0
    same_length = 1
    while len(walk) < L_max:
        # 筛选出可以前往的邻居节点
        n_neighbors = list(G.neighbors(node))
        # 不存在邻居则直接结束
        if len(n_neighbors) == 0:
            break
        V_stay = [x for x in n_neighbors if x[0] == node[0]]
        # 计算Pr跳转概率
        if len(V_stay) == 0:
            Pr_stay = 0
        elif len(n_neighbors) == len(V_stay):
            Pr_stay = 1
        else:
            Pr_stay = math.pow(a, same_length)

        # (0,1)均匀抽样，决定JUMP or STAY
        r = random.uniform(0, 1)
        if r<=Pr_stay:
            # Stay
            random_node = random.choice(V_stay)
        else:
            # JUMP
            # 首先选取jump类型
            Q_jump = list(set([x[0] for x in n_neighbors if (x[0] not in Q_hist)]))
            if len(Q_jump)>0:
                # 选取JUMP节点
                jump_type = random.choice(Q_jump)
                V_jump = [x for x in n_neighbors if x[0] == jump_type]
                random_node = random.choice(V_jump)
            else:
                Q_jump = list(set([x[0] for x in n_neighbors if (x[0] != node[0])]))
                jump_type = random.choice(Q_jump)
                V_jump = [x for x in n_neighbors if (x[0] == jump_type)]
                random_node = random.choice(V_jump)
        if random_node[0] not in Q_hist:
            Q_hist.append(random_node[0])
            if len(Q_hist)>m:
                Q_hist.pop(0)
        elif random_node[0] == Q_hist[-1]:
            same_length+=1

        # 添加路径节点
        walk.append(random_node)

        if node[0]!=random_node[0]:
            same_length = 1

        node = random_node
    return walk

def Just_walks(G,N_walk,L_max,a):
    walks = []
    nodes = list(G.nodes())
    print('Generating walks .. ')
    for node in tqdm(nodes):
        for i in range(N_walk):
            just_walks = get_justwalk(G, node, L_max,a)
            walks.append(just_walks)
    print('Generating finished. ')
    return walks

def walk_training(walks,window_size,dimensions,output_file):   
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, workers=-1)
    model.wv.save_word2vec_format(output_file)
    return model

if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    G = nx.read_edgelist(args.input_file)
    print('Starting training .. ')
    walks = Just_walks(G, args.num_walks, args.walk_length, args.alpha)
    walk_training(walks,args.window_size,args.dimensions, args.output_file)
    t2 = time.time()
    print('Finished. Total Time :', t2-t1)