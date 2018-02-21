import os
import numpy as np

files = os.listdir('../../../../corpora/adaptive/replacement_simplicity/surveymonkey/')

map = {}
for file in files:
	data = file.strip().split('_')
	group = data[2]
	cands = data[3]

	if group not in map:
		map[group] = []

	f = open('../../../../corpora/adaptive/replacement_simplicity/surveymonkey/'+file)
	for line in f:
		linedata = line.strip().split('\t')
		score = float(linedata[1])
		if cands=='10':
			map[group].append(score)
	f.close()

for group in sorted(map.keys()):
	print group, ': ', np.mean(map[group])
