import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

import scipy.stats as ss

def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))
		

def load_ontology(file_name, genes):
	gene2id_mapping = { genes[i] : i for i in range(0, len(genes) ) }

	dG = nx.DiGraph()
	term_direct_gene_map = {}
	term_size_map = {}

	file_handle = open(file_name)

	gene_set = set()

	for line in file_handle:
		line = line.rstrip().split()
		
		if line[2] == 'default':
			dG.add_edge(line[0], line[1])
		else:
			if line[1] not in gene2id_mapping:
				continue

			if line[0] not in term_direct_gene_map:
				term_direct_gene_map[ line[0] ] = set()

			term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

			gene_set.add(line[1])

	file_handle.close()

	print('There are', len(gene_set), 'genes')

	for term in dG.nodes():
		
		term_gene_set = set()

		if term in term_direct_gene_map:
			term_gene_set = term_direct_gene_map[term]

		deslist = nxadag.descendants(dG, term)

		for child in deslist:
			if child in term_direct_gene_map:
				term_gene_set = term_gene_set | term_direct_gene_map[child]

		# jisoo
		if len(term_gene_set) == 0:
			print('There is empty terms, please delete term:', term)
			sys.exit(1)
		else:
			term_size_map[term] = len(term_gene_set)

	leaves = [n for n,d in dG.in_degree() if d==0]

	uG = dG.to_undirected()
	connected_subG_list = list(nxacc.connected_components(uG))

	print('There are', len(leaves), 'roots:', leaves[0])
	print('There are', len(dG.nodes()), 'terms')
	print('There are', len(connected_subG_list), 'connected componenets')

	if len(leaves) > 1:
		print('There are more than 1 root of ontology. Please use only one root.')
		sys.exit(1)
	if len(connected_subG_list) > 1:
		print('There are more than connected components. Please connect them.')
		sys.exit(1)

	return dG, leaves[0], term_size_map, term_direct_gene_map


def load_data(file_name):
	feature = []
	label = []

	with open(file_name, 'r') as fi:
		for line in fi:
			tokens = line.strip().split('\t')

			flist = list(map(float, tokens[0].split(',')))
			feature.append(flist)
			label.append([float(tokens[1])])

	return feature, label


# load mppaing table into a list
def load_mapping(mapping_file):

	mapping = []

	file_handle = open(mapping_file)

	for line in file_handle:
		line = line.rstrip().split()
		mapping.append(line[1])

	file_handle.close()
	
	return mapping


# load one data file
def prepare_1dataset(data_file):
    
	test_feature, test_label = load_data(data_file)
	return (torch.Tensor(test_feature), torch.Tensor(test_label))


# load two data files
def prepare_2datasets(train_file, test_file):

	# load mapping files
	train_feature, train_label = load_data(train_file)
	test_feature, test_label = load_data(test_file)

	return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label))


# build feacture vectors for each batch
# row_data: indices of data points
def build_input_vector(inputdata, num_col, original_features):

	num_row = inputdata.shape[0]
	cuda_features = torch.zeros(num_row, num_col, dtype=torch.half) 

	for i in range(num_row):
		gene1 = int(inputdata[i, 0])
		gene2 = int(inputdata[i, 1])
	
		if gene1 != gene2:
			cuda_features.data[i] = original_features.data[gene1] + original_features.data[gene2]
		else:
			sys.stderr.write("The input is single gene knock out.\n")
			cuda_features.data[i] = original_features.data[gene1]

	return cuda_features


