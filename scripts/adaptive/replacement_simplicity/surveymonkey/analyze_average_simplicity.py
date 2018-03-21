import os
import numpy as np
from scipy.stats import ttest_ind

files = os.listdir('../../../../corpora/adaptive/replacement_simplicity/surveymonkey/')

map = {}
for file in files:
	data = file.strip().split('_')
	ranker = data[0]
	group = data[2]
	cands = data[3]

	if group not in map:
		map[group] = {}
	if ranker not in map[group]:
		map[group][ranker] = []

	f = open('../../../../corpora/adaptive/replacement_simplicity/surveymonkey/'+file)
	for line in f:
		linedata = line.strip().split('\t')
		score = float(linedata[1])
		if cands=='10':
			map[group][ranker].append(score)
	f.close()

print '\t'.join(sorted(map[group].keys()))
for group in sorted(map.keys()):
	newline = group + ': '
	for ranker in sorted(map[group].keys()):
		newline += '\t' + str(np.mean(map[group][ranker]))
	print newline

for ranker in sorted(map[group].keys()):
	groups = sorted(map.keys())
	for i in range(0, len(groups)-1):
		for j in range(i+1, len(groups)):
			t, p = ttest_ind(map[groups[i]][ranker], map[groups[j]][ranker])
			print ranker, ': ', groups[i], groups[j], ' - ', p
