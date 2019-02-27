import os
import re
import numpy as np
import pickle 
import random
import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

from tsne import tsne,pca
from nltk.stem import WordNetLemmatizer



# split action labels into words
def label_to_tokens(l):
	return l.strip().split(' ')

# cleaning words. 
# remove special characers and space.
# convert to lower case
def clean_word(word):	
	return re.sub('[^A-Za-z0-9]+', '', word.strip().lower())

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

# construct a vocab
def prepare_vocab(alabels):

	vocab = []
	for (db, labels) in alabels.items():	
		for l in labels:
			vocab += label_to_tokens(l)

	# cleaning vocab and vocab = list(set(vocab))
	vocab = [clean_word(v) for v in vocab]

	match = 0
	unmatched_words = []
	for w in vocab:
		if w in w2vdic_words:
			match +=1
		else:
			unmatched_words.append(w)

	print 'List of unmatched words:', unmatched_words
	print 'percentage of matched words', match / (1.0*len(vocab))

	return vocab

#prepare_vocab(action_labels)


# get word embeddings
def token_to_vec(t):

	# list from https://gist.github.com/sebleier/554280
	stop_words = ["i", "me", "my", "myself", "we", "our", "ours", 
			"ourselves", "you", "your", "yours", "yourself", 
			"yourselves", "he", "him", "his", "himself", 
			"she", "her", "hers", "herself", "it", 
			"its", "itself", "they", "them", "their", 
			"theirs", "themselves", "what", "which", "who", 
			"whom", "this", "that", "these", "those", "am", 
			"is", "are", "was", "were", "be", "been", "being", 
			"have", "has", "had", "having", "do", "does", "did", 
			"doing", "a", "an", "the", "and", "but", "if", "or", 
			"because", "as", "until", "while", "of", "at", "by", 
			"for", "with", "about", "against", "between", "into", 
			"through", "during", "before", "after", "above", "below", 
			"to", "from", "up", "down", "in", "out", "on", "off", "over", 
			"under", "again", "further", "then", "once", "here", "there", 
			"when", "where", "why", "how", "all", "any", "both", "each", 
			"few", "more", "most", "other", "some", "such", "no", "nor", 
			"not", "only", "own", "same", "so", "than", "too", "very", 
			"s", "t", "can", "will", "just", "don", "should", "now"]

	t = clean_word(t)
	flag = 0
	vec = np.zeros((1, emb_dim))		 

	if len(t) < 3:
		return flag, vec 
	
	# dont consider stop words
	if t in stop_words:
		return flag, vec

	# apply stemming
	if _lemmatize: 
		t = lm.lemmatize(t, pos='v')
		t = str(t)

	if t in w2vdic_words:		
		vec = w2vdic[t]
		flag = 1
		
	#print t, vec
	return flag, vec	
 
# get sum label embedding
# not average.
def label_to_vec(l):
	count = 0
	vec = np.zeros((1, emb_dim))
	tokens = label_to_tokens(l)	
	for t in tokens:
		flag, rv = token_to_vec(t)		
		vec = vec + rv
		count = count + flag
	#print l, 1/(1.0*len(tokens)) * vec

	#if count == 0:
	#	return vec
	#else:
	#	return 1/(1.0*count) * vec 

	return count, vec

## get average db embedding
def db_to_vec(db):
     
     db_labels = action_labels[db]
     vec = np.zeros((1, emb_dim))

     #if len(db_labels) > 50:
	#db_labels = random.sample(db_labels, 50)

     db_wordcount = 0
     for l in db_labels:
	label_wc, v = label_to_vec(l)
	db_wordcount += label_wc
	vec += v
  	
     return 1/(1.0*db_wordcount) * vec


def run_tsne(X,d, p):	
	_map = tsne(X,2, d, p)
	return _map


# plot 2d image
def show_2dimage(Y, label_names, class_names, colors):

	font_size, marker_size = 6, 10 #10, 10

	# remove spaces at the boundaries if any
	label_names = [l.strip() for l in label_names]
	class_names = [c.strip() for c in class_names]		
	class_names_unique = set(class_names)
	
	fig, ax = plt.subplots()

	# when plotting large number of points
	if len(label_names)>100:
		font_size, marker_size = 3, 6	# decrease font and marker size
		Y = 10.*Y # stretch the plot

	for (cn,cl) in zip(class_names_unique, colors):
		idx = [i for i, x in enumerate(class_names) if x == cn]		
	    	ax.scatter(Y[idx,0], Y[idx,1], s=marker_size, c=cl, label=cn)

	if level == 'action':
		ax.legend(class_names_unique, loc='best', fontsize=3)

    	for i, txt in enumerate(label_names):
        	ax.annotate(txt, (Y[i,0], Y[i,1]), fontsize=font_size) #, rotation=random.choice(range(0,90,30)))
     	
	fig.savefig('2d.eps', format= 'eps', dpi=1000)
	plt.close(fig)
	
   	return



## main function
if __name__ == "__main__":

	# define some global variables
	np.random.seed(19)
	level = 'action' #'dataset' #'action'
	_lemmatize = True #True
	label_file = './action_labels.txt'
	action_labels = {}
	embedding_fn = "glove.6B.50d.txt"
	emb_dim = 50

	feat_cache_fn = 'feat_cache_' + level + '_lemma_' + str(_lemmatize)
	
	if os.path.exists(feat_cache_fn):
                #features, feature_labels= np.load('feat_cache.npy')
		with open(feat_cache_fn) as f:
		   features, feature_labels, feature_db = pickle.load(f)	
	else:

		print('Obtaining word embeddings..')
		lm = WordNetLemmatizer()
		with open(embedding_fn, "rb") as lines:
		    w2vdic = {line.split()[0]: np.array(map(float, line.split()[1:]))
			   for line in lines}
		w2vdic_words = w2vdic.keys()

		# read label files and create a dictionary
		total_actions = 0 
		with open(label_file, 'r') as f:
		    for l in f:
			line = l.strip().split(',')
			#remove empty labels
			line = [x for x in line if x]
			action_labels[line[0]] = line[1:]	
			total_actions = total_actions + len(line[1:])

		
		if level == 'dataset':
			i = 0
			features = np.zeros((len(action_labels.keys()), emb_dim))	
			feature_labels = []
			for (db,labels) in sorted(action_labels.items()):
				features[i] = db_to_vec(db)
				features[i] = features[i] /(1.0*np.linalg.norm(features[i],2))
				feature_labels.append(db)		
				i = i + 1
			feature_db = feature_labels

		if level == 'action':
			features = []
			feature_labels = []
			feature_db = []
			i = 0
			for (db,labels) in sorted(action_labels.items()):

				#for charades and something, show only 5
				if db == 'something something' or db == 'charades':
					labels = labels[:5]

				if len(labels) > 50:
					labels = random.sample(labels, 50)
		
				for l in labels:
					if not is_ascii(l):
					    continue
					l = l.strip().lower()
					c, label_fea = label_to_vec(l) # gives sum of feas
					if c>0: 
					     label_fea = label_fea/(1.*c)  # get average
					if np.sum(label_fea) == 0:
						continue	
					if l in feature_labels:
						continue
					features.append(label_fea)
					feature_labels.append(l)
					feature_db.append(db)
					#print db, i, l.strip().lower()
					features[i] = features[i] /(1.0*np.linalg.norm(features[i],2))
					i = i + 1                				
		
			features = np.asarray(features).squeeze()
		
		with open(feat_cache_fn, 'wb') as f:
    			pickle.dump((features, feature_labels, feature_db), f)
		

	if level == 'dataset':
		pca_dim = 5 
		perplexity = 3

		# ucf101 and thumos are same, so one less dimension
		pca_dim = len(feature_db) - 1
		perplexity = 2
	else:
		pca_dim = 50
		perplexity = 5

	# colors to encoude points for each db. make sure it is same as number of db
	# bit lazy to generate automatically with cmap
	colors = ['red','green','blue', 'cyan', 'magenta', 'yellow', 'black',  
			'brown', 'salmon', 'violet', 'pink', 'khaki', 
			'olive', 'silver', 'tan', 'lavender', 'maroon']
        
	assert len(colors) == len(list(set(feature_db)))		
	
	# run tsne
	Y = run_tsne(features, pca_dim, perplexity)
	#Y = pca(features, 2).real

	# plot on 2d graph and save the image
	show_2dimage(Y, feature_labels, feature_db, colors)
