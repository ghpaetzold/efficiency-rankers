import matplotlib.pyplot as plt
import os, numpy, scipy

def produceMap(path):
	map = {}
	files = os.listdir(path)
	for file in files:
		info = file.strip().split('_')
		ranker = info[0]
		dataset = info[1]
		group = info[2]
		prop = info[3]
		if ranker not in map:
			map[ranker] = {}
		if group not in map[ranker]:
			map[ranker][group] = {}
		map[ranker][group][prop] = []
		f = open(path+'/'+file)
		for line in f:
			data = line.strip().split('\t')
			index = int(data[0])
			score = float(data[1])
			map[ranker][group][prop].append([index, score])
	return map
	
def getBaselineMap(path):
	map = {}
	files = os.listdir(path)
	for file in files:
		info = file.strip()
		info = info[:len(info)-4].split('_')
		dataset = info[0]
		ranker = info[1]
		if dataset not in map:
			map[dataset] = {}
		f = open(path+'/'+file)
		scores = [float(v) for v in f.readline().strip().split('\t')]
		f.close()
		map[dataset][ranker] = scores
	return map

def getDatasetSizes():
	map = {}
	map['benchls'] = 929
	map['nnseval'] = 239
	map['semeval'] = 2010
	return map

names = {}
names['benchls'] = 'BenchLS'
names['semeval'] = 'SemEval'
names['boundary'] = 'Boundary'
names['ridge'] = 'Regression'
names['mlp'] = 'Regression'
names['normal'] = 'Baseline'
names['individual'] = 'ID'
names['group-age'] = 'Age'
names['group-edu'] = 'Education'
names['group-lang'] = 'Native Language'
names['group-langg'] = 'Language Group'
names['group-langgg'] = 'Language Family'
names['group-prof'] = 'Proficiency'

performancesno = '../../../corpora/adaptive/simplicity_accuracy/surveymonkey/'
mapno = produceMap(performancesno)
performancesyes = '../../../corpora/adaptive/profile_as_features/surveymonkey/'
mapyes = produceMap(performancesyes)

mapnames = ['Without features', 'With features']

allgroups = ['normal','individual','group-age','group-edu', 'group-lang', 'group-langg', 'group-langgg', 'group-prof']

#Create figure
fig, axes = plt.subplots(len(mapno['ridge']), len(mapno),figsize=(14,20))
for i, group in enumerate(allgroups):
	yvalues = []
	ranker = 'ridge'
	for j, vmap in enumerate([mapno, mapyes]):
		colors = [(c, c, c) for c in [v/100.0 for v in range(10, 70, 5)]]
		for prop in vmap[ranker][group]:
			seq = vmap[ranker][group][prop]
			x = [value[0] for value in seq]
			y = [value[1] for value in seq]
			yvalues.extend(y)
			axes[i][j].plot(x, y, color=colors.pop())
			axes[i][j].title.set_text(names[group] + ' - ' + mapnames[j])
			axes[i][j].set_ylim([0.45,0.75])
			axes[i][j].set_yticks([0.45, 0.55, 0.65, 0.75])
			axes[i][j].grid(linestyle='-.')
	
plt.tight_layout(pad=0.4, w_pad=0.8, h_pad=1.0)
plt.savefig('profile_as_features_results.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()