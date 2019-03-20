"""
Rianne Lyons
LING 539 Assignment 3
Filename: problem3.py
Last modified: 26 November 2017

Implementation of a visible Markov model POS tagger using the Viterbi algorithm 
at inference and add-one smoothing for the probabilities.
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
	with open('model3.pkl', 'wb') as model:
		pickle.dump(matrix1, model)
		pickle.dump(matrix2, model)
		pickle.dump(tagcount, model)
		pickle.dump(vsize, model)
		pickle.dump(tsize, model)
	
	return

"""
Loads a saved model named "model3.pkl" and assigns its contents to dictionaries.
Returns: mat1 and mat2 are dictionaries holding the loaded content, voc is the size of the 
         vocabulary, tags is the number of tags, counttags is the dictionary of tags and their
         counts.
"""	
def load_model():
	import pickle
	with open('model3.pkl', 'rb') as m:
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
		if word_tag[1] == 'START' or word_tag[1] == 'END':
			continue
		else:
			word_probabilities[word_tag] = float(wcount[word_tag] + 1.0) / float(tcount[word_tag[1]] + len(w))	
	
	save_model(tag_probabilities, word_probabilities, len(w), t, tcount)	
	return

"""
Checks the accuracy of the predicted tags for a given sentence, calculating the number of 
correct overall tags and the number of correct tags for unknown words.
Parameters: pred_tags is the list of predicted tags, tags is the list of correct tags, and ukns 
            is a dictionary where the time step is the key and the value is 0 if the word is known 
            and 1 if the word is unknown.
Returns: a tuple of the overall correct tags and the correct unknown tags for the given sentence.
"""	
def check_accuracy(pred_tags, tags, ukns):
	correct = 0
	ukn_correct = 0
	for r in range(len(pred_tags)):
		if pred_tags[r] == tags[r]:
			correct += 1
			if ukns[-1*(r-len(pred_tags))] == 1:
				ukn_correct += 1
				continue
			else:
				continue
	return (correct, ukn_correct)
	
"""
Tests the model on the given testing data using the Viterbi algorithm and add-one smoothing. 
Calculates accuracy for all words and for unknown words.
Parameters: test_file is the file containing the test_data, a_mat is the tag transition matrix, 
            b_mat is the emission matrix, states is the list of tags, tagcounts is the dictionary 
            of tags and their counts, voc_size is the length of the vocabulary.
"""	
def test(test_file, a_mat, b_mat, states, tagcounts, voc_size):
	viterbi = {}
	backpointer = {}
	path = []
	total_tags = 0
	count_correct = 0
	debug_count = 0
	ukn_count = 0
	ukn_words = 0
	states.remove('END')
	states.remove('START')
	
	with open(test_file, 'r') as f:
		sent = []
		track_ukn = {}
		#initialization step
		tstep = 0
		for s in states:
			try: 
				a = a_mat['START ' + s]
			except KeyError:
				a = 1.0 / float(tagcounts[s] + len(states))
			b = 1.0
			viterbi[s] = [float(a)*float(b)]
			backpointer[s] = ['start']
				
		for line in f:
			if line == '\n' or line == '\t\n':
				#termination
				maxv = 0.0
				argterm_state = ''
				for s in viterbi.keys():
					try:
						a = a_mat[s + ' END']
					except KeyError:
						a = 1.0 / float(tagcounts[s] + len(states))
					v = viterbi[s][-1]*a
					if v > maxv:
						maxv = v
						argterm_state = s
				st = argterm_state	
				for r in range(tstep, 0, -1):
					path.append(st)
					st = backpointer[st][r]
				rev = path[::-1]
				total_tags += len(sent)
				corr, ucorr = check_accuracy(rev, sent, track_ukn)
				count_correct += corr
				ukn_count += ucorr
				#initialization
				del sent[:]
				del path[:]
				viterbi.clear()
				backpointer.clear()
				track_ukn.clear()
				tstep = 0
				for s in states:
					try: 
						a = a_mat['START ' + s]
					except KeyError:
						a = 1.0 / float(tagcounts[s] + len(states))
					b = 1.0
					viterbi[s] = [float(a)*float(b)]
					backpointer[s] = ['start']	
				continue
			else: #time step 
				word, tab, temp = line.partition('\t')
				tag = temp.rstrip('\n')
				tstep += 1
				is_ukn = 0
				sent.append(tag)
				for s in states:
					m = 0.0
					argstate = ''
					for s_prime in viterbi.keys():
						v = viterbi[s_prime][tstep-1]
						try:
							a = a_mat[s_prime + ' ' + s]
						except KeyError:
							a = 1.0 / (float(tagcounts[s] + len(states)))
						try:
							b = b_mat[(word, s)]
							is_ukn = 0
						except KeyError:
							b = 1.0 / (float(tagcounts[s] + voc_size))
							ukn_words += 1
							is_ukn = 1
						vit = float(v)*float(a)
						if vit > m:
							m = vit
							argstate = s_prime
					viterbi[s].append(m*float(b))
					backpointer[s].append(argstate)
					track_ukn[tstep] = is_ukn
								
	total_accuracy = 100*(float(count_correct)/float(total_tags))
	print 'total accuracy: ', total_accuracy, '%' #88.86%
	ukn_accuracy = 100*(float(ukn_count)/float(ukn_words))
	print 'unknown word accuracy: ', ukn_accuracy, '%' #0.05%
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
		tag_prob_matrix, word_prob_matrix, voc_len, tags_list, tag_nums = load_model()
		print 'testing'
		test(test_fname, tag_prob_matrix, word_prob_matrix, tags_list, tag_nums, voc_len)
	else:
		print "Please specify 'train' or 'test'"

	return
	
if __name__ == '__main__':
	main()

