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
		score = float(f.readline().strip())
		map[dataset][ranker] = score
	return map

performancesf = '../corpora/performances'
map = produceMap(performancesf)
baselinesf = '../corpora/baselines'
base = getBaselineMap(baselinesf)


for dataset in map:
	for ranker in map[dataset]:
		colors = list('bgrcmyk')
		plt.clf()
		plt.title(ranker)
		for prop in map[dataset][ranker]:
			seq = map[dataset][ranker][prop]
			x = [value[0] for value in seq]
			y = [value[1] for value in seq]
			a, b = numpy.polyfit(numpy.log(x), y, 1)
			plt.plot(x, y, colors.pop()+'-')
			#plt.plot(x, a*numpy.log(x)+b, 'g-')
			#plt.axis([0, 6, 0, 20])
		for ranker in base[dataset]:
			plt.axhline(y=base[dataset][ranker], color='b')
		plt.show()