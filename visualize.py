
label_file = './action_labels.txt'
action_labels = {}

# read label files and create a dictionary 
with open(label_file, 'r') as f:
    for l in f:
	line = l.strip().split(',')
	#remove empty labels
	line = [x for x in line if x]
	action_labels[line[0]] = line[1:]
	
print action_labels


