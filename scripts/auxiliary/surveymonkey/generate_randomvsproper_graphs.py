import matplotlib.pyplot as plt
import os, numpy, scipy
from scipy.stats import *

def produceMap(path):
	map = {}
	files = os.listdir(path)
	for file in files:
		info = file.strip().split('_')
		ranker = info[0]
		dataset = info[1]
		group = info[2][:-4]
		if ranker not in map:
			map[ranker] = {}
		if group not in map[ranker]:
			map[ranker][group] = {}
		map[ranker][group]['normal'] = []
		map[ranker][group]['random'] = []
		f = open(path+'/'+file)
		for line in f:
			data = line.strip().split('\t')
			normalv = float(data[0].strip())
			randomv = float(data[1].strip())
			map[ranker][group]['normal'].append(normalv)
			map[ranker][group]['random'].append(randomv)
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

performancesf = '../../../corpora/adaptive/random_vs_proper/surveymonkey/'
map = produceMap(performancesf)

allgroups = ['individual','group-age','group-edu', 'group-lang', 'group-langg', 'group-langgg', 'group-prof']

#Create figure
fig, axes = plt.subplots(len(map['ridge'])-1, len(map),figsize=(7,20))
for i, group in enumerate(allgroups):
	for j, ranker in enumerate(['boundary','ridge']):
			t, prob = ttest_ind(map[ranker][group]['random'], map[ranker][group]['normal'])
			print group, ranker, prob
			values = [map[ranker][group]['random'], map[ranker][group]['normal']]
			axes[i][j].boxplot(values, vert=False, whis=89999)
			axes[i][j].title.set_text(names[ranker]+' - '+names[group])
			axes[i][j].set_xlim([0.45,0.75])
			axes[i][j].set_xticks([0.45, 0.55, 0.65, 0.75])
			axes[i][j].grid(linestyle='-.')
plt.setp(axes, yticks=[1, 2], yticklabels=['R', 'N'])
		
#plt.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.0)
plt.savefig('randomvsnormal_results.png', dpi=150)
#plt.savefig('customized_results_new.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()