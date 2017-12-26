import matplotlib.pyplot as plt
import os, numpy, scipy

def produceMap(path):
	map = {}
	files = os.listdir(path)
	for file in files:
		info = file.strip().split('_')
		ranker = info[0]
		dataset = info[1]
		prop = info[2]
		if dataset not in map:
			map[dataset] = {}
		if ranker not in map[dataset]:
			map[dataset][ranker] = {}
		map[dataset][ranker][prop] = []
		f = open(path+'/'+file)
		for line in f:
			data = line.strip().split('\t')
			index = int(data[0])
			score = float(data[1])
			map[dataset][ranker][prop].append([index, score])
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
names['ridge'] = 'Ridge'
names['mlp'] = 'Neural'
	
performancesf = '../../corpora/performances'
map = produceMap(performancesf)
baselinesf = '../../corpora/baselines'
base = getBaselineMap(baselinesf)
sizes = getDatasetSizes()

for dataset in map:
	for ranker in map[dataset]:
		colors = list('bgrcmykbgrcmyk')
		plt.figure(figsize=(10,5))
		plt.title('Results for the '+names[ranker]+' ranker on the '+names[dataset]+' dataset')
		plt.xlabel('Number of training instances')
		plt.ylabel('Pearson correlation')
		for prop in ['0.2', '0.4', '0.6', '0.8']: #map[dataset][ranker]:
			seq = map[dataset][ranker][prop]
			x = [value[0] for value in seq]
			y = [value[1] for value in seq]
			# plt.plot(x, y, 'k-')
			plt.plot(x, y, colors.pop()+'-')
		# for ranker in base[dataset]:
			# y = base[dataset][ranker]
			# x = [int(sizes[dataset]*float(z)/100.0) for z in range(100, 0, -10)]
			# plt.plot(x, y, color='k')
		plt.savefig(dataset+'-'+ranker+'.png', dpi=300, bbox_inches='tight')
		plt.show()
		

#a, b = numpy.polyfit(numpy.log(x), y, 1)
#plt.plot(x, a*numpy.log(x)+b, 'g-')
#plt.axis([0, 6, 0, 20])

#plt.axhline(y=base[dataset][ranker], color='b')