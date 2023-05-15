import sys
import os
import numpy as np
from numpy import genfromtxt
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from dcell import *
import argparse
import time



# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		# adjust precision
		mask = torch.zeros(len(gene_set), gene_dim, dtype=torch.half)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

		term_mask_map[term] = mask_gpu

	return term_mask_map



# function to train a DCell model
def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, test_data, gene_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, gene_features, opt_level, keep_batchnorm_fp32, loss_scale, CUDA_ID):

	torch.cuda.set_device(CUDA_ID)

	# load training data
	train_feature, train_label = train_data

	# add data labels to CUDA memory
	train_label_gpu = Variable(train_label.cuda(CUDA_ID))

	# load cell line mutation profiles and NW embedding for KO genes to CPU
	cuda_genes = torch.from_numpy(gene_features)

	# define model and add to CUDA memory
	model = dcell(term_size_map, term_direct_gene_map, dG, gene_dim, root, num_hiddens_genotype, CUDA_ID)
	model.cuda(CUDA_ID)

	# define ADAM optimizer: stochastic gradient descent algorithm
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
	
	# set gradients to zero for correct parameter updates
	optimizer.zero_grad()

	# for all nodes in a hierarchy, create a mask vector to turn off all irrelevant genes
	term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)


	# change initial model parameters
	for name, param in model.named_parameters():
		term_name = name.split('_')[0]

		if '_direct_gene_layer.weight' in name:
			param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			#param.data = torch.mul(param.data, term_mask_map[term_name])
		else:
			param.data = param.data * 0.1

	# define data loader for training data
	train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size=batch_size, shuffle=False)

	# record current time to measure time elapsed
	start_time = time.time()


	# start model training
	for epoch in range(train_epochs):

		# let the model know it is in training phase: BatchNorm, Dropout layer would behave accordingly.
		model.train()
		train_predict = torch.zeros(0,0).cuda(CUDA_ID)

		# logging elapsed time for each batch
		batchstart_time = time.time()

		# repeat for each batch
		for i, (inputdata, labels) in enumerate(train_loader):

			cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

			# Forward + Backward + Optimize
			optimizer.zero_grad()  # zero the gradient buffer

			# run forward function
			# aux_out_map is a dictionary: term -> function output
			aux_out_map, _ = model(inputdata)

			# collect prediction for the training data
			if train_predict.size()[0] == 0:
				train_predict = aux_out_map[root].data
			else:
				train_predict = torch.cat([train_predict, aux_out_map[root].data], dim=0)

			# calculate total_loss
			total_loss = 0
			for name, output in aux_out_map.items():

				# Utilize different loss functions depending on the problem
				loss = nn.MSELoss()
				if name == root:
					total_loss += loss(output, cuda_labels)
					#print ('=======', total_loss.dtype)
				else: # change 0.2 to smaller one for big terms
					total_loss += 0.2 * loss(output, cuda_labels)
			
			total_loss.backward()

			# remove irrelevant gene data
			for name, param in model.named_parameters():
				if '_direct_gene_layer.weight' not in name:
					continue
				term_name = name.split('_')[0]
				param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

			# update parameters based on current gradient
			optimizer.step()

			# logging elapsed time for each batch
			batchend_time = time.time()
			sys.stderr.write("Batch %d elapsed time: %s seconds\n" % (i, (batchend_time-batchstart_time)))
			batchstart_time = batchend_time
			
			# clean up memory
			torch.cuda.empty_cache()
			del param.grad


		# put model to evaluation mode and calculate the performace of the model on training data
		model.eval()
		train_corr = pearson_corr(train_predict, train_label_gpu)

		# save models
		# uncomment the if statement to speed up
		#if epoch % 10 == 0:
		checkpoint = { 'model': model.state_dict(), 'optimizer': optimizer.state_dict() }
		torch.save(checkpoint, model_save_folder + '/model_' + str(epoch))


		test_corr = -1
		# evaluate model using validation set
		# can be omitted to speed up ###################################################################
		if len(test_data) > 0:
			test_feature, test_label = test_data
			test_label_gpu = Variable(test_label.cuda(CUDA_ID))

			test_loader = du.DataLoader(du.TensorDataset(test_feature, test_label), batch_size=batch_size, shuffle=False)
			test_predict = torch.zeros(0, 0).cuda(CUDA_ID)
			total_loss = 0
		
			for i, (inputdata, labels) in enumerate(test_loader):

				cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				# run forward function
				# aux_out_map is a dictionary: term -> function output
				aux_out_map, _ = model(inputdata)

				# collect prediction for the training data
				if test_predict.size()[0] == 0:
					test_predict = aux_out_map[root].data
				else:
					test_predict = torch.cat([test_predict, aux_out_map[root].data], dim=0)

				# calculate total_loss
				total_loss += loss(aux_out_map[root].data, cuda_labels)
		
			test_corr = pearson_corr(test_predict, test_label_gpu)
		##################################################################################################


		# measure time elapsed
		end_time = time.time()		
		print("Epoch\t%d\tCUDA_ID\t%d\ttrain_corr\t%.6f\ttest_corr\t%.6f\ttotal_loss\t%.6f\telapsed_time\t%s seconds" % (epoch, CUDA_ID, train_corr, test_corr, total_loss, end_time-start_time))
		start_time = end_time

		

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train dcell')

	parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
	parser.add_argument('-train', help='Training dataset', type=str)
	parser.add_argument('-test', help='Validation dataset', type=str, default='none')
	parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
	parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
	parser.add_argument('-batchsize', help='Batchsize', type=int, default=10000)
	parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
	parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)

	parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
	parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=5)


	# call functions
	opt = parser.parse_args()
	torch.set_printoptions(precision=5)

	# load input data
	train_data = prepare_1dataset(opt.train)

	test_data = []
	if opt.test != "none":
		test_data = prepare_1dataset(opt.test)

	mutgenes = load_mapping(opt.gene2id)
	num_genes = len(mutgenes)

	# load cell features: binary matrix (mutation profile)
	gene_features = np.identity(num_genes)

	# load ontology
	dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, mutgenes)

	# load the number of hiddens #######
	num_hiddens_genotype = opt.genotype_hiddens

	# specify GPU
	CUDA_ID = opt.cuda

	train_model(root, term_size_map, term_direct_gene_map, dG, train_data, test_data, num_genes, opt.modeldir, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, gene_features, CUDA_ID)

