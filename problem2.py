"""
Rianne Lyons
LING 539 Assignment 3
Filename: problem2.py
Last modified: 26 November 2017

Implementation of a visible Markov model POS tagger, using add-one smoothing
and the greedy inference strategy.
Requires pickle to save and load the model.
To train and test, respectively:
python2 problem2.py train train.tagged
python2 problem2.py test test.tagged
"""

"""
Read in training data and build counts and lists of words, tags, and tag transitions.
Parameters: f is the file containing the training data.
Returns: tag_counts is a dictionary of tag: count, bigram_counts is a dictionary of bigram:count, 
         word_tag_counts is a dictionary of (word, tag): count, and tag_list, word_list, and 
         bigram_list are lists of tags, words, and bigrams, where bigrams are the jth tag followed
         by the kth tag.
"""
def setup(f):
	tag_counts = {} # #of occurrences of each tag
	bigram_counts = {} # # of occurrences of a tag followed by another tag
	word_tag_counts = {} # # of occurrences of a word tagged with a tag
	bigram_list = []
	tag_list = []
	word_list = []
	
	with open(f, 'r') as fname:
		l = fname.readline()
		word, tab, t = l.partition('\t')
		tag = t.rstrip('\n')
		tag_temp = 'START'
		word_temp = 'START'
		bigram = tag_temp + ' ' + tag
		word_tag_counts[(word, tag)] = 1
		word_tag_counts[(word_temp, tag_temp)] = 1
		tag_counts[tag_temp] = 1
		tag_counts[tag] = 1
		tag_list.append(tag)
		tag_list.append(tag_temp)
		word_list.append(word)
		word_list.append(word_temp)
		bigram_list.append(bigram)
		
		for line in fname:
			if line == '\n':
				word = 'END'
				tag = 'END'
				bigram = tag_temp + ' ' + tag
				tag_temp = 'START'
				word_temp = 'START'
			else:
				word, tab, t = line.partition('\t')
				tag = t.rstrip('\n')
				bigram = tag_temp + ' ' + tag
				tag_temp = tag
				word_temp = word
			if word not in word_list:
				word_list.append(word)
			if tag not in tag_list:
				tag_list.append(tag)
			if bigram not in bigram_list:
				bigram_list.append(bigram)
			if (word, tag) in word_tag_counts.keys():
				word_tag_counts[(word, tag)] += 1
			else:
				word_tag_counts[(word, tag)] = 1
			if tag in tag_counts.keys():
				tag_counts[tag] += 1
			else:
				tag_counts[tag] = 1
			if bigram in bigram_counts.keys():
				bigram_counts[bigram] += 1
			else:
				bigram_counts[bigram] = 1
		word = 'END'
		tag = 'END'
		bigram = tag_temp + ' ' + tag
		if bigram not in bigram_list:
			bigram_list.append(bigram)
		if (word, tag) in word_tag_counts.keys():
			word_tag_counts[(word, tag)] += 1
		else:
			word_tag_counts[(word, tag)] = 1
		if tag in tag_counts.keys():
			tag_counts[tag] += 1
		else:
			tag_counts[tag] = 1
		if bigram in bigram_counts.keys():
			bigram_counts[bigram] += 1
		else:
			bigram_counts[bigram] = 1
			
	return (tag_counts, bigram_counts, word_tag_counts, tag_list, word_list, bigram_list)

"""
Saves the matrices from training using pickle. 
Parameters: matrix1 and matrix2 are the matrices to write, vsize is the size of the vocabulary, 
            tsize is the number of tags, and tagcount is the dictionary of tags with their counts.
"""
def save_model(matrix1, matrix2, vsize, tsize, tagcount):
	import pickle
	with open('model2.pkl', 'wb') as model:
		pickle.dump(matrix1, model)
		pickle.dump(matrix2, model)
		pickle.dump(tagcount, model)
		pickle.dump(vsize, model)
		pickle.dump(tsize, model)
		
	return

"""
Loads a saved model named "model2.pkl" and assigns its contents to dictionaries.
Returns: mat1 and mat2 are dictionaries holding the loaded content, voc is the size of the 
         vocabulary, tags is the number of tags, counttags is the dictionary of tags and their
         counts.
"""	
def load_model():
	import pickle
	with open('model2.pkl', 'rb') as m:
		mat1 = pickle.load(m)
		mat2 = pickle.load(m)
		counttags = pickle.load(m)
		voc = pickle.load(m)
		tags = pickle.load(m)
		
	return (mat1, mat2, voc, tags, counttags)

"""
Builds the model, which consists of two matrices, one of tag transition probabilities and one 
of emission probabilities. Probabilities are calculated using add-one smoothing. Saves the model.
Parameters: b is the list of tag transitions, tcount is the dictionary of tags and their counts, 
            bcount is the dictionary of tag transitions and their counts, wcount is the 
            dictionary of emissions and their counts, w is the list of the vocabulary, and t is 
            the list of tags.
"""
def train(b, w, t, tcount, bcount, wcount):
	tag_probabilities = {'END START': 1.0}
	word_probabilities = {}
	
	for tag_bigram in b:
		#tag_bigram represents second tag given first tag
		jth_tag, space, kth_tag = tag_bigram.partition(' ')
		tag_probabilities[tag_bigram] = float(bcount[tag_bigram] + 1.0) / float(tcount[jth_tag] + len(t))
	
	for word_tag in wcount.keys():
		word_probabilities[word_tag] = float(wcount[word_tag] + 1.0) / float(tcount[word_tag[1]] + len(w))	
	
	save_model(tag_probabilities, word_probabilities, len(w), len(t), tcount)	
		
	return

"""
Tests the model on the given testing data using the greedy inference strategy. Computes accuracy 
for total words and for unknown words, using add-one smoothing for unknown words.
Parameters: test_file is the file containing the test data, tag_probs is the tag transition 
            matrix, word_probs is the emission probability matrix, vocab_size is the size of the 
            vocabulary, tag_size is the number of tags, and tag_nums is the dictionary of tags and 
            their counts.
"""
def test(test_file, tag_probs, word_probs, vocab_size, tag_size, tag_nums):
	total_tags = 0
	num_correct = 0
	total_ukn = 0
	correct_ukn = 0
	
	with open(test_file, 'r') as f:
		likelihoods = {}
		results = []
		priors = {}
		prev_tag = 'START'
		
		for line in f:
			del results[:]
			likelihoods.clear()
			priors.clear()
			if line == '\n' or line == '\t\n':
				prev_tag = 'START'
				continue
			else:
				word, tab, t = line.partition('\t')
				tag = t.rstrip('\n')
				total_tags += 1
				for tagtrans in tag_probs.keys(): #build priors dict
					kth_tag, s, ith_tag = tagtrans.partition(' ')
					if prev_tag == kth_tag:
						if ith_tag not in priors.keys():
							priors[ith_tag] = tag_probs[tagtrans]
						else:
							if tag_probs[tagtrans] > priors[ith_tag]:
								priors[ith_tag] = tag_probs[tagtrans]
				for wordtag in word_probs.keys(): #word(i) | tag(i), build likelihoods dict
					if word == wordtag[0]:
						likelihoods[wordtag[1]] = word_probs[wordtag] #tag: prob
				if len(likelihoods.keys()) > 0: #known word
					for l in likelihoods.keys():
						if l in priors.keys():
							results.append((l, float(likelihoods[l])*float(priors[l])))
						else: 
							p = 1.0 / (float(tag_nums[prev_tag] + tag_size))
							results.append((l, float(likelihoods[l]*p)))
					results.sort(key=lambda result: result[1])
					pred_tag = results[-1][0]
					if pred_tag == tag:
						num_correct += 1
				else:
					total_ukn += 1
					
					for pr in priors.keys():
						lh = 1.0 / (float(tag_nums[pr] + vocab_size))
						results.append((pr, float(lh)*float(priors[pr])))
					results.sort(key=lambda result: result[1])
					pred_tag = results[-1][0]
					if pred_tag == tag:
						num_correct += 1
						correct_ukn += 1
					
				prev_tag = pred_tag	
		
	ukn_accuracy = 100*(float(correct_ukn)/float(total_ukn))
	total_accuracy = 100*(float(num_correct)/float(total_tags))
	print 'total accuracy: ', total_accuracy, '%' #90.49%
	print 'unknown word accuracy: ', ukn_accuracy, '%' #27.70%
	
	return

"""
Main function to control flow of actions specified when executing the program. If the program 
is executed with 'train' specified, a model will be trained on training data and saved; if the 
program is executed with 'test' specified, the trained model will be loaded and tested on 
testing data.
"""
def main():
	import sys
	action = sys.argv[1]
	if action == 'train':
		fname = sys.argv[2]
		print 'setting up'
		ctags, cbigrams, cwords, tags, words, bigrams = setup(fname)
		print 'training'
		train(bigrams, words, tags, ctags, cbigrams, cwords)
	elif action == 'test':
		test_fname = sys.argv[2]
		print 'loading model'
		tag_prob_matrix, word_prob_matrix, voc_len, tags_len, tag_nums = load_model()
		print 'testing'
		test(test_fname, tag_prob_matrix, word_prob_matrix, voc_len, tags_len, tag_nums)
	else:
		print "Please specify 'train' or 'test'"
		
	return
	
if __name__ == '__main__':
	main()

