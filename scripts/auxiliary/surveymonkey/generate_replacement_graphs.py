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

performancesf = '../../../corpora/adaptive/replacement_accuracy/surveymonkey/'
map = produceMap(performancesf)
baselinesf = '../../../corpora/baselines'
#base = getBaselineMap(baselinesf)
#sizes = getDatasetSizes()

allgroups = ['normal','individual','group-age','group-edu', 'group-lang', 'group-langg', 'group-langgg', 'group-prof']

#Create figure
fig, axes = plt.subplots(len(map['ridge']), len(map),figsize=(14,20))
for i, group in enumerate(allgroups):
	yvalues = []
	for j, ranker in enumerate(['boundary','ridge']):
	#for j, ranker in enumerate(['ridge']):
		#colors = [(c, c, c) for c in [v/100.0 for v in range(5, 50, 5)]]
		colors = list('bgrcmbgrcmbgrcmbgrcmbgrcm')
		#fig.suptitle('Comparison between Boundary and Neural rankers')
#		for prop in ['0.2', '0.4', '0.6', '0.8']:
		for prop in map[ranker][group]:
			seq = map[ranker][group][prop]
			x = [value[0] for value in seq]
			y = [value[1] for value in seq]
			yvalues.extend(y)
			axes[i][j].plot(x, y, color=colors.pop())
			axes[i][j].title.set_text(names[ranker]+' - '+names[group])
			axes[i][j].set_ylim([0.05,0.35])
			axes[i][j].set_yticks([0.05, 0.15, 0.25, 0.35])

#plt.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.8, h_pad=1.0)
#plt.savefig('customized_results.png', dpi=150)
plt.savefig('replacement_accuracy_results.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()