#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import networkx as nx
import random
import math
from gensim.models import Word2Vec
from tqdm import tqdm
import os

def get_het_dic(DATA):
	if DATA=='DBLP':
		heterg_dictionary={'a': ['p'], 'p': ['a', 't', 'v'], 't': ['p'], 'v': ['p']}
	elif DATA=='Movie':
		heterg_dictionary={'d': ['m'], 'm': ['a', 'd', 'c'], 'a': ['m'], 'c': ['m']}
	elif DATA=='Foursquare':
		heterg_dictionary={'t': ['c'], 'c': ['u', 't', 'p'], 'u': ['c'], 'p': ['c']}
	else:
		print('Datasets error')
	return heterg_dictionary

def parse_args():
	parser = argparse.ArgumentParser(description="Just")

	parser.add_argument('--input',default='Datasets/DBLP/dblp.edgelist')
    
	parser.add_argument('--DATA',default='DBLP')

	parser.add_argument('--dimensions' , type=int,default=128)

	parser.add_argument('--walk_length' , type=int,default=100)

	parser.add_argument('--num_walks' , type=int,default=1)

	parser.add_argument('--window-size' , type=int,default=10)

	parser.add_argument('--alpha' , type=float,default=0.5)

	parser.add_argument('--workers' , default=10)

	parser.add_argument('--train' , default=1)#just walks
    
	parser.add_argument('--memory' , default=2)#memory domain length
    
	parser.add_argument('--output',default='EmbeddingData')
    
	return parser.parse_args()

def dblp_generation(G, path_length, heterg_dictionary, m, start):#生成一条just walks
	path = []   
	path.append(start)
	homog_length = 1#同类点走的长度即L
	no_next_types = 0    
	heterg_probability = 0
	memory_domain=[]
	while len(path) < path_length:
		#heterg_dictionary={'a': ['p'], 'p': ['a', 't', 'v'], 't': ['p'], 'v': ['p']}
		heterg_dictionary=get_het_dic(args.DATA)
		if no_next_types == 1:
			break
		cur = path[-1]#获得上一个节点 node_type,node_name       
		homog_type = []
		heterg_type = []                
		for node_type in heterg_dictionary:#同异字典 key=node_type value=hete_type
			if cur[0] == node_type:
				homog_type = node_type
				heterg_type = heterg_dictionary[node_type]
#		print homog_type,heterg_type,cur[0],heterg_dictionary
		if not heterg_type:
			break
		if homog_type not in memory_domain:
			if len(memory_domain)<m:
				memory_domain.append(homog_type)
			else:
				memory_domain.pop(0)
				memory_domain.append(homog_type)
		else:
			memory_domain.remove(homog_type)
			memory_domain.append(homog_type)
            
		heterg_probability = 1 - math.pow(args.alpha, homog_length)#1-P
		r = random.uniform(0, 1)
		next_type_options=[]#下一个节点name，不是种类
#——————————————————————————————————————
		if r <= heterg_probability:#Jump
        
			temp=heterg_type[:]#save heterg_typr，跳不出记忆域也要跳
			for item in memory_domain:
				if item in heterg_type:
					heterg_type.remove(item)                   
			for heterg_type_iterator in heterg_type:
				next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
#                
			if not next_type_options:
				for heterg_type_iterator in temp:
					next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
#                
			if not next_type_options:#没异边走，继续走同边
				next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
			if not next_type_options:
				break
#——————————————————————————————————————
		else:#Stay
			next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
			if not next_type_options:#如果没同边则走异边
				for heterg_type_iterator in heterg_type:
					next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
		if not next_type_options:#如果下一步选项为空则break
			no_next_types = 1
			break
			
		next_node = random.choice(next_type_options)
		path.append(next_node)
		if next_node[0] == cur[0]:
			homog_length = homog_length + 1
		else:
			 homog_length = 1
	return path

def generate_walks(G, num_walks, walk_length, m, heterg_dictionary):
	print('Generating walks .. ')
	walks = []
	nodes = list(G.nodes())
	for cnt in tqdm(range(num_walks)):
		random.shuffle(nodes)
		for node in nodes:
			just_walks = dblp_generation(G, walk_length, heterg_dictionary, m, start=node)
			walks.append(just_walks)
	print('Walks done .. ')
	return walks

def main(args):
	G = nx.read_edgelist(args.input)
	heterg_dictionary=get_het_dic(args.DATA)
	walks = generate_walks(G, args.num_walks, args.walk_length, args.memory, heterg_dictionary)
	print('Finished walks .. ')
	np.save(os.path.join(args.output,args.DATA+'walks.npy'),walks)    
	if args.train:
		print('Starting training .. ')
		model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, workers=args.workers)
		print('Finished training .. ')
		model.wv.save_word2vec_format(os.path.join(args.output,args.DATA+'.embeddings'))

		keys=list(model.wv.vocab.keys())
		wordvector=[]
        #save as npy file to ML
		for key in keys:
			wordvector.append(model[key])
		np.save(os.path.join(args.output,args.DATA+'_w2v_embedding.npy'),wordvector)
		np.save(os.path.join(args.output,args.DATA+'_w2v_id.npy'),keys)


if __name__ == "__main__":
	args = parse_args()
	main(args)




