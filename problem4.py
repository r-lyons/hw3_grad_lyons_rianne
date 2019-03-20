"""
Rianne Lyons
LING 539 Assignment 3
Filename: problem3.py
Last modified: 26 November 2017

Implementation of an RNN with GRUs using DyNet for POS tagging. 
"""

import numpy as np
import pickle
import dynet_config
dynet_config.set(mem=700, autobatch=True, random_seed=1978)
import dynet as dy

"""
Read in training and test data and build lists of sentences.
Parameters: train_fname is the file of training data, testf is the file
            of testing (or development) data.
Returns: train_word_seq is a list of sentences from training, train_tag_seq is the list
         of tags from training, tag_map assigns each tag an index, idx_map reverses tag_map, 
         test_word_seq is a list of sentences from training, test_tag_seq is the list of those 
         sentences' tags.
"""
def setup(train_fname, testf):
	train_word_seq = []
	train_tag_seq = []
	temp_sent = []
	temp_tags = []
	test_word_seq = []
	test_tag_seq = []
	temp_s = []
	temp_t = []
	tag_map = {}
	idx_map = {}
	vocab = []
	
	with open(train_fname, 'r') as f:
		
		for line in f:
			if line == '\n' or line == '\t\n':
				train_word_seq.append(temp_sent)
				train_tag_seq.append(temp_tags)
				temp_sent = []
				temp_tags = []
				continue
				
			else:
				w, s, t = line.partition('\t')
				tag = t.rstrip('\n')
				word = w.lower()
				temp_sent.append(word)
				temp_tags.append(tag)
				if word not in vocab:
					vocab.append(word)
				if tag not in tag_map.keys():
					tag_map[tag] = len(tag_map.keys())
				continue
	
	train_tag_seq = [[tag_map[t] for t in tseq] for tseq in train_tag_seq]
	for tg in tag_map.keys():
		idx_map[tag_map[tg]] = tg
		
	with open(testf, 'r') as f:
		for line in f:
			if line == '\n' or line == '\t\n':
				test_word_seq.append(temp_s)
				test_tag_seq.append(temp_t)
				temp_s = []
				temp_t = []
				continue
			else:
				word, s, t = line.partition('\t')
				tag = t.rstrip('\n')
				temp_s.append(word)
				temp_t.append(tag)
				continue
	
	test_tag_seq = [[tag_map[t] for t in tseq] for tseq in test_tag_seq]
		
	return (train_word_seq, train_tag_seq, tag_map, idx_map, test_word_seq, test_tag_seq)

"""
Load the first 10000 word embeddings from a given file and put them into a matrix.
Parameters: embed_file is the file containing the embeddings, vec_size is the dimension (300 here).
Returns: word_idx is a dictionary mapping the word from the embedding to its index in the matrix, 
         embed_matrix is the matrix of embeddings.
"""
def import_embeddings(embed_file, vec_size):
	embed_matrix = []
	word_idx = {}
	with open(embed_file, 'r') as f:
		embed_matrix.append(np.array([0.0 for r in range(vec_size)]))
		word_count = 1
		for line in f:
			if word_count == 10001:
				break
			else:
				l = line.rstrip('\n').split(' ')
				word_idx[l[0]] = word_count
				embed_matrix.append(np.array([float(e) for e in l[1:]]))
				word_count += 1
	
	return (word_idx, np.matrix(embed_matrix))		

"""
For a given word sequence, converts each word to its pre-assigned integer index.
Parameters: word_seq is the sequence to convert, lookup_mat contains the index for each word.
Returns: idx_seq is the sequence of indexes corresponding to the input sequence of words.
"""
def convert_wordidx(word_seq, lookup_mat):
	idx_seq = []
	for w in word_seq:
		idx = lookup_mat.get(w.lower(), 0) #return 0 if unknown word
		idx_seq.append(idx)
	return idx_seq

"""
Run through a forward pass with a given model and input.
Parameters: embed_params is the matrix of parameter embeddings, gru_unit is a GRU RNN unit, gru_model 
            is the RNN model, wd_idx is the mapping of words to indexes, param_mat is the matrix of hidden layer parameters, param_bias is the parameter bias, xin is the input of indices.
Returns: gru_out is the result of the forward pass.
"""
def set_model_forward(embed_params, gru_unit, gru_model, wd_idx, param_mat, param_bias, xin):
	xinput = [embed_params[x] for x in xin]
	pmatrix_exp = dy.parameter(param_mat)
	bias_exp = dy.parameter(param_bias)
	gru_seq = gru_unit.initial_state()
	gru_hidden = gru_seq.transduce(xinput)
	gru_out = [dy.transpose(pmatrix_exp) * l + bias_exp for l in gru_hidden]
	return gru_out

"""
Uses the weights produced for the tags to get the indices of the predicted tags with a softmax layer.
Parameters: weights is the list of weights for each tag.
Returns: tag_idx is the list of predicted tag indices (the argmax of the weights for each tag).
"""
def get_pred_tags(weights):
	sm_preds = [dy.softmax(w) for w in weights]
	sm_to_np = [s.npvalue() for s in sm_preds]
	tag_idx = [np.argmax(n) for n in sm_to_np]
	return tag_idx

"""
Trains the model for a given number of epochs.
Parameters: epochs is the number of epochs to run, em_params is the matrix of embedding parameters, 
            gunit is the RNN unit, index_tg is the mapping from indices to tags, mdl is the RNN model, 
            train_wds is the list of training sentences, train_tgs is the list of corresponding tags, 
            word_idx is the mapping from words to indices, pmat is the hidden layer parameter, pbias is 
            the bias parameter.
Returns: batch is the batch size, mdl is the model.
"""	
def training(epochs, em_params, gunit, index_tg, mdl, train_wds, train_tgs, word_idx, pmat, pbias):
	import random as rd
	sgd_train = dy.SimpleSGDTrainer(m=mdl, learning_rate=0.01)
	batch = 40
	train_batches = int(np.ceil(float(len(train_wds)) / float(batch)))
	loss_per_epoch = []
	
	for epoch_idx in range(epochs):
		eloss = []
		rd.seed(epoch_idx)
		rd.shuffle(train_wds)
		rd.seed(epoch_idx)
		rd.shuffle(train_tgs)
		for batch_idx in range(50):
			dy.renew_cg()
			batch_wds = train_wds[batch_idx*batch:(batch_idx+1)*batch]
			batch_tags = train_tgs[batch_idx*batch:(batch_idx+1)*batch]
			for sent_idx in range(len(batch_wds)):
				idxs = convert_wordidx(batch_wds[sent_idx], word_idx)
				forwardp = set_model_forward(em_params, gunit, mdl, word_idx, pmat, pbias, idxs)
				wd_losses = [dy.pickneglogsoftmax(forwardp[p], batch_tags[sent_idx][p]) for p in range(len(idxs))]
				sentence_losses = dy.esum(wd_losses)
				sentence_losses.backward()
				sgd_train.update()
				eloss.append(sentence_losses.npvalue())
		loss_per_epoch.append(np.sum(eloss))
		
	return batch, mdl

"""
Computes the accuracy of all the predicted tags, overall and for unknown words.
Parameters: predictions is the list of predicted tag indices, labels is the list of actual tag indices, 
            id_tg is the mapping from indices to tags, indexes is the list of word indices (holding 0 
            for unknown words).
Returns: total_accuracy is the overall accuracy, ukn_accuracy is accuracy on unknown words.
"""
def check_accuracy(predictions, labels, id_tg, indexes):
	total_labels = 0
	num_correct = 0
	total_ukn = 0
	ukn_correct = 0
	print 'checking accuracy'
	for r in range(len(predictions)):
		for p in range(len(predictions[r])):
			total_labels += 1			
			if predictions[r][p] == labels[r][p]:
				num_correct += 1
				if indexes[r][p] == 0:
					total_ukn += 1
					ukn_correct += 1
				else:
					continue
			else:
				if indexes[r][p] == 0:
					total_ukn += 1
					continue
				else:
					continue
	total_accuracy = float(num_correct) / float(total_labels)
	ukn_accuracy = float(ukn_correct) / float(total_ukn)			
	return (total_accuracy, ukn_accuracy)

"""
Tests the model on a given test (or development) set and checks the prediction accuracy for the document.
Paramters: index_tg is the mapping from index to tag, gu is the RNN unit, m is the RNN model, len_batch 
           is the batch size, test_wds is the list of test sentences, test_tgs are the corresponding 
           tags, wd_idx is the mapping from word to index, pmat is the hidden layer parameter, pbis is 
           the bias parameter, em_params is the matrix of embedding parameters.
"""
def testing(index_tg, gu, m, len_batch, test_wds, test_tgs, wd_idx, pmat, pbias, em_params):
	doc_preds = []
	ids = []
	test_batches = int(np.ceil(float(len(test_wds)) / float(len_batch)))
	for batch_idx in range(50):	
		dy.renew_cg()
		batch_wds = test_wds[batch_idx*len_batch:(batch_idx+1)*len_batch]
		batch_tgs = test_tgs[batch_idx*len_batch:(batch_idx+1)*len_batch]
		for sent_idx in range(len(batch_wds)):	
			idxs = convert_wordidx(batch_wds[sent_idx], wd_idx)
			forwardp = set_model_forward(em_params, gu, m, wd_idx, pmat, pbias, idxs)
			sent_preds = get_pred_tags(forwardp)
			doc_preds.append(sent_preds)
			ids.append(idxs)
	t, u = check_accuracy(doc_preds, test_tgs, index_tg, ids)
	print 'total accuracy: ', t*100, '%'
	print 'unknown word accuracy: ', u*100, '%'
	return 
	
"""
Main function to control flow of actions specified when executing the program. If the program 
is executed with 'train' specified, a model will be trained; if 'tune' is specified, training is 
repeated with a varying number of epochs and the dev set is used to test; if 'test' is specified, 
a model is trained with the best number of epochs and tested on the test set. Model saving 
functionality is commented out due to a significant drop in accuracy when loading the model.
"""
def main():
	import sys
	action = sys.argv[1]
	emb_name = sys.argv[2]
	filen = sys.argv[3]
	testname = sys.argv[4]
	
	print 'setting up'
	train_words, train_tags, tag_idx, idx_tag, test_words, test_tags = setup(filen, testname)
	num_tags = len(tag_idx.keys())
	gru_model = dy.ParameterCollection()
	word_index, embeddings_mat = import_embeddings(emb_name, 300)
	hidden_layer_len = 200
	layers = 1 
	eparams = gru_model.lookup_parameters_from_numpy(embeddings_mat.A) #flatten matrix
	gru_unit = dy.GRUBuilder(layers, 300, hidden_layer_len, gru_model)
	param_mat = gru_model.add_parameters((hidden_layer_len, num_tags))
	param_bias = gru_model.add_parameters((num_tags))
	#gmodel.save("grumodel.model")
	#mdl2 = dy.ParameterCollection()
	#ep = mdl2.lookup_parameters_from_numpy(embeddings_mat.A)
	#parmat = mdl2.add_parameters((200, num_tags))
	#parbias = mdl2.add_parameters((num_tags))
	#gmodel.populate("grumodel.model")
	
	if action == 'train':
		print 'training'
		bsize, gmodel = training(3, eparams, gru_unit, idx_tag, gru_model, train_words, train_tags, word_index, param_mat, param_bias)
	if action == 'tune':
		print 'tuning'
		for r in range(3, 6):
			print 'training'
			print 'epochs: ', r
			bsize,gmodel = training(r, eparams, gru_unit, idx_tag, gru_model, train_words, train_tags, word_index, param_mat, param_bias)
			print 'testing'
			testing(idx_tag, gru_unit, gmodel, bsize, test_words, test_tags, word_index, param_mat, param_bias, eparams)
	if action == 'test':
		print 'training'
		#use 5 epochs
		bsize, gmodel = training(5, eparams, gru_unit, idx_tag, gru_model, train_words, train_tags, word_index, param_mat, param_bias)
		print 'testing'
		testing(idx_tag, gru_unit, gmodel, bsize, test_words, test_tags, word_index, param_mat, param_bias, eparams)

	return
	
if __name__ == '__main__':
	main()
