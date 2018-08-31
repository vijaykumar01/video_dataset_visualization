import numpy as np
import re
import random
import matplotlib
matplotlib.use('Agg')
from tsne import tsne
import pylab
import matplotlib.pyplot as plt


def label_to_tokens(l):
	return l.strip().split(' ')

# cleaning words
def clean_word(word):	
	return re.sub('[^A-Za-z0-9]+', '', word.strip().lower())

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

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

def token_to_vec(t):

	t = clean_word(t)
	flag = 0
	vec = np.zeros((1,50))		 

	if len(t) < 3:
		return flag, vec 

	if t in w2vdic_words:		
		vec = w2vdic[t]
		flag = 1
		
	#print t, vec
	return flag, vec	
 
def label_to_vec(l):
	count = 0
	vec = np.zeros((1, 50))
	tokens = label_to_tokens(l)	
	for t in tokens:
		flag, rv = token_to_vec(t)		
		vec = vec + rv
		count = count + flag
	#print l, 1/(1.0*len(tokens)) * vec
	if count == 0:
		return vec
	else:
		return 1/(1.0*count) * vec 

def db_to_vec(db):
     db_labels = action_labels[db]
     #if len(db_labels) > 50:
	#db_labels = random.sample(db_labels, 50)
     vec = np.zeros((1, 50))
     for l in db_labels:
	vec += label_to_vec(l)
     return 1/(1.0*len(db_labels)) * vec


def run_tsne(X,d, p):	
	_map = tsne(X,2, d, p)
	return _map

def show_2dimage(Y, label_names, class_names, colors):

	font_size, marker_size = 10, 10

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

	ax.legend(class_names_unique, loc='best', fontsize=3)

    	for i, txt in enumerate(label_names):
        	ax.annotate(txt, (Y[i,0], Y[i,1]), fontsize=font_size) #, rotation=random.choice(range(0,90,30)))
     	
	fig.savefig('2d.eps', format= 'eps', dpi=1000)
	plt.close(fig)
	
   	return


## main function
label_file = './action_labels.txt'
action_labels = {}

with open("glove.6B.50d.txt", "rb") as lines:
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
        #if line[0] == 'Hollywood2' or line[0] == 'ucf101' or line[0] == 'thumos':
        #if not (line[0] == 'sports-1m' or line[0] == 'youtube-8m' or line[0] == 'kinetics'):
        action_labels[line[0]] = line[1:]	
	total_actions = total_actions + len(line[1:])

level = 'action' #'dataset'
if level == 'dataset':
	i = 0
	features = np.zeros((len(action_labels.keys()), 50))	
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

		#if db == 'sports-1m' or db == 'youtube-8m':
		#	continue
		if len(labels) > 50:
			labels = random.sample(labels, 50)
		
		for l in labels:
			if not is_ascii(l):
			    continue
			l = l.strip().lower()
			label_fea = label_to_vec(l)
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
			#if i >1000:
			#  break			
		
	features = np.asarray(features).squeeze()
	

#dist = np.sum((features[None,:] - features[:, None])**2, -1)**0.5	
np.savetxt('fea.txt',features,fmt='%f',delimiter=',')
np.savetxt('fea_labels.txt', feature_labels, fmt='%s')
np.savetxt('fea_db.txt', feature_db, fmt='%s')
#np.savetxt('sim.txt',dist,fmt='%f', delimiter=',')

if level == 'dataset':
	pca_dim = 5 
	perplexity = 3 
else:
	pca_dim = 50
	perplexity = 5

# colors to encoude points for each db. make sure it is same as number of db
# bit lazy to generate automatically with cmap
colors = ['red','green','blue', 'cyan', 'magenta', 'yellow', 'black',  'brown', 'salmon', 'violet', 'pink', 'khaki', 'olive', 'silver', 'tan']

# run tsne
Y = run_tsne(features, pca_dim, perplexity)

# plot on 2d graph and save the image
show_2dimage(Y, feature_labels, feature_db, colors)
