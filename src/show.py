import numpy as np
import networkx as nx
import random
import math
import time
import argparse
from tqdm import tqdm
import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


random.seed(0)

#@st.cache(persist=True,show_spinner=False)
def get_justwalk(G,node,L_max,a,m=2):
    walk = [node]
    Q_hist = [node[0]]
    Pr_stay = 0
    same_length = 1
    strategy = []
    current_strategy = {'Q_jump':[node[0]],}
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
        current_strategy['Pr_stay'] = Pr_stay
        # (0,1)均匀抽样，决定JUMP or STAY
        r = random.uniform(0, 1)
        if r<=Pr_stay:
            # Stay
            current_strategy['JUST'] = 'STAY'
            random_node = random.choice(V_stay)
        else:
            # JUMP
            current_strategy['JUST'] = 'JUMP'
            # 首先选取jump类型
            Q_jump = list(set([x[0] for x in n_neighbors if (x[0] not in Q_hist)]))
            current_strategy['Q_jump'] = Q_jump
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
        current_strategy['Q_hist'] = Q_hist.copy()
        current_strategy['next_domain'] = random_node[0]
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
        strategy.append(current_strategy.copy())
    return walk,strategy

def Just_walks(G,N_walk,L_max,a):
    walks = []
    nodes = list(G.nodes())
    print('Generating walks .. ')
    for node in tqdm(nodes):
        for i in range(N_walk):
            just_walks,strategy= get_justwalk(G, node, L_max,a)
            walks.append(just_walks)
    print('Generating finished. ')
    return walks

def walk_training(walks,window_size,dimensions,output_file):   
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, workers=-1)
    model.wv.save_word2vec_format(output_file)
    return model

def get_graph(G,walks,L):
    edges = list(G.edges)
    edge_list = [x for x in list(G.edges) if (x[0]==walks[-1] or x[1]==walks[-1])]
    if L==len(walks):
        edge_list = []
    for i in range(len(walks)-1):
        edge_list.append((walks[i],walks[i+1]))
    # st.write(edge_list)
    G_show = nx.Graph()
    G_show.add_edges_from(edge_list)
    color = ['yellow' for x in range(len(G_show.nodes))]
    data = pd.DataFrame({'node':G_show.nodes,'color':color})
    data = data.set_index('node')
    data.loc[walks] = 'orange'
    return G_show,data

st.title('JU&ST strategy')

st.sidebar.title('Parameter setting')
G = nx.read_edgelist('../Datasets/DBLP/dblp.edgelist')
node = 'p0'
just_walks,strategy= get_justwalk(G, node, L_max = 10,a=0.5)

if st.sidebar.button('Random start node'):
    node = random.choice(list(G.nodes))
    st.sidebar.text('The node we start: %s'%(node))
    just_walks,strategy= get_justwalk(G, node, L_max = 10,a=0.5)



walk = [node]

L = st.sidebar.slider('Choose steps',value=1,min_value = 1,max_value = len(just_walks)-1)
st.sidebar.text('前进到第%s步'%L)

G_show,data = get_graph(G,just_walks[:L],len(just_walks))
#just_walks[:L]


s = strategy[L-1]
st.write('Stay probability:',s['Pr_stay'] ,' Next select by probability is ',s['JUST'])
st.write('Q_hist',str(s['Q_hist']),'\tPossible domain:',str(s['Q_jump']) ,'Next select domain',s['next_domain'])


fig, ax = plt.subplots()
p = nx.spring_layout(G_show)
for i in range(len(G_show.nodes)):
    p[i] = [-10*i,0]
nx.draw(G_show ,ax =ax, pos = p, node_color=data.color.tolist(),
    edge_color='#000000',width=1, node_size = 500,edge_cmap=plt.cm.gray, 
    with_labels=True)
plt.show()
st.write(fig)


# 源数据查看选项
agree = st.checkbox('查看生成路径')
if agree:
    st.write(just_walks)
# 代码展示
agree1 = st.checkbox('查看源代码')
if agree1:
    f = open('main_show.py',encoding='UTF-8')
    st.code(f.read(), language='python')

