import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from dcell import *
import argparse



def predict_model(root, term_size_map, term_direct_gene_map, dG, test_data, gene_dim, model_file, train_epochs, batch_size, learning_rate, num_hiddens_genotype, gene_features, opt_level, keep_batchnorm_fp32, loss_scale, CUDA_ID, hidden_dir, result_dir):

	device = torch.device("cuda:%d" % CUDA_ID)
	torch.cuda.set_device(CUDA_ID)	

	model_id = model_file[model_file.rfind('/')+1:]

	model = dcell(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, CUDA_ID)
	model.cuda(CUDA_ID)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
	checkpoint = torch.load(model_file, map_location=device)

	model.eval()

	# test data
	test_feature, test_label = test_data
	test_label_gpu = Variable(test_label.cuda(CUDA_ID))

	# load cell line mutation profiles and NW embedding for KO genes to CPU
	cuda_genes = torch.from_numpy(gene_features)

	# define data loader for testing data
	test_loader = du.DataLoader(du.TensorDataset(test_feature, test_label), batch_size=batch_size, shuffle=False)

	test_predict = torch.zeros(0,0).cuda(CUDA_ID)
	term_hidden_map = {}	

	for i, (inputdata, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		cuda_gene_features = build_input_vector(inputdata, gene_dim, cuda_genes)

		# make prediction for test data
		aux_out_map, term_hidden_map = model(cuda_gene_features, cuda_drug_features)

		# collect prediction made by model
		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		# print out values of hidden variable
		for term, hidden_map in term_hidden_map.items():
			this_hidden_file = hidden_dir+'/'+term+'_'+str(i)+'.txt'
			hidden_file = hidden_dir+'/'+term+'.hidden'

			np.savetxt(this_hidden_file, hidden_map.data.cpu().numpy(), '%.4e')	
	
			# append the file to the main file
			os.system('cat ' + this_hidden_file + ' >> ' + hidden_file)
			os.system('rm ' + this_hidden_file)


	test_corr = pearson_corr(test_predict, test_label_gpu)
	#print 'Test pearson corr', model.root, test_corr	
	print("Model\t%s Test pearson corr\t%s\t%.6f" % (model_id, model.root, test_corr))
	print("")

	np.savetxt(result_dir + '/' + model_id + '.predict', test_predict.cpu().numpy(),'%.4e')

	del param.grad
	torch.cuda.empty_cache()



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Test DrugCell model')

	parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
	parser.add_argument('-test', help='Validation dataset', type=str)
	parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
	parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
	parser.add_argument('-batchsize', help='Batchsize', type=int, default=10000)
	parser.add_argument('-model', help='Path to the trained models', type=str)
	parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)

	parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
	parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=5)

	parser.add_argument('-hidden', help='Hidden output folder', type=str, default='Hidden/')
	parser.add_argument('-result', help='Result file name', type=str, default='Result/')

	opt = parser.parse_args()
	torch.set_printoptions(precision=5)

	# load input data
	test_data = prepare_1dataset(opt.test)
	mutgenes = load_mapping(opt.gene2id)
	num_genes = len(mutgenes)

	# load cell features: binary matrix (mutation profile)
	gene_features = np.genfromtxt(opt.cellline, delimiter=",", dtype=np.float16)

	# load ontology
	dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, mutgenes)

	# load the number of hiddens #######
	num_hiddens_genotype = opt.genotype_hiddens

	# specify GPU: default is 0
	CUDA_ID = opt.cuda

	predict_model(root, term_size_map, term_direct_gene_map, dG, test_data, num_genes, opt.model, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, gene_features, CUDA_ID, opt.hidden, opt.result)
