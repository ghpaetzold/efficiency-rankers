import sys, random
import numpy as np
from scipy.stats import *
from sklearn import linear_model
from lexenstein.features import *
from sklearn.cross_validation import train_test_split

def toText(data):
	result = '\n'.join(['\t'.join(line) for line in data])
	return result

def mountInstances(path, mode):
	map = {}
	size = 0
	name = 'colingsr'
	f = open(path)
	insts = []
	for line in f:
		data = line.strip().split('\t')
		id = data[0]
		age = int(float(data[1])/10.0)
		lang = data[2]
		edu = data[3]
		prof = data[4]
		keymap = {'age':age, 'lang':lang, 'edu':edu, 'prof':prof}
		sent = data[5]
		target = data[6]
		sent = sent.replace('_____________', target)
		index = data[7]
		cands = set([data[8], data[9]])
		first = data[10]
		cands.remove(first)
		second = list(cands)[0]

		inst = [sent, target, index, '1:'+first, '2:'+second]
		if mode=='normal':
			key = 'all'
		elif mode=='individual':
			key = id
		elif mode.startswith('group'):
			group = mode.split('-')[1].strip()
			key = keymap[group]
		else:
			key = 'all'
		if key not in map:
			map[key] = []
		map[key].append(inst)
		size += 1
	f.close()
	return map, name, size

class OnlineRegressionRanker:

	def __init__(self, fe):
		self.fe = fe

	def trainRegressionModel(self, training_data_text):
		#Create matrix:
		features = self.fe.calculateFeatures(training_data_text, format='victor', input='text')
		Xtr = []
		Ytr = []
		c = -1
		for line in training_data_text.strip().split('\n'):
			data = line.strip().split('\t')
			cands = [cand.strip().split(':')[1] for cand in data[3:]]
			indexes = [int(cand.strip().split(':')[0]) for cand in data[3:]]
			featmap = {}
			for cand in cands:
				c += 1
				featmap[cand] = features[c]
			for i in range(0, len(cands)-1):
				for j in range(i+1, len(cands)):
					indexi = indexes[i]
					indexj = indexes[j]
					indexdiffji = indexj-indexi
					indexdiffij = indexi-indexj
					positive = featmap[cands[i]]
					negative = featmap[cands[j]]
					v1 = np.concatenate((positive,negative))
					v2 = np.concatenate((negative,positive))
					Xtr.append(v1)
					Xtr.append(v2)
					Ytr.append(indexdiffji)
					Ytr.append(indexdiffij)
		Xtr = np.array(Xtr)
		Ytr = np.array(Ytr)

		self.model = linear_model.Ridge()
		self.model.fit(Xtr, Ytr)
		return self.model

	def getRankings(self, data):
		#Transform data:
		textdata = ''
		for inst in data:
			for token in inst:
				textdata += token+'\t'
			textdata += '\n'
		textdata = textdata.strip()

		#Create matrix:
		features = self.fe.calculateFeatures(textdata, input='text')

		ranks = []
		c = -1
		for line in data:
			cands = [cand.strip().split(':')[1].strip() for cand in line[3:]]
			featmap = {}
			scoremap = {}
			for cand in cands:
				c += 1
				featmap[cand] = features[c]
				scoremap[cand] = 0.0
			for i in range(0, len(cands)-1):
				cand1 = cands[i]
				for j in range(i+1, len(cands)):
					cand2 = cands[j]
					posneg = np.concatenate((featmap[cand1], featmap[cand2]))
					probs = self.model.predict(np.array([posneg]))
					score = probs[0]
					scoremap[cand1] += score
					negpos = np.concatenate((featmap[cand2], featmap[cand1]))
					probs = self.model.predict(np.array([negpos]))
					score = probs[0]
					scoremap[cand1] -= score
			rank = sorted(scoremap.keys(), key=scoremap.__getitem__, reverse=True)
			if len(rank)>1:
				if rank[0]==line[1].strip():
					rank = rank[1:]
			ranks.append(rank)
		return ranks

class RankerEvaluator:

	def evaluateRanker(self, gold_data, rankings):
		#Read data:
		index = -1
		all_gold = []
		all_ranks = []
		for line in gold_data:
			index += 1
			gold_rankings = {}
			for subst in line[3:len(line)]:
				subst_data = subst.strip().split(':')
				word = subst_data[1].strip()
				ranking = int(subst_data[0].strip())
				gold_rankings[word] = ranking
			ranked_candidates = rankings[index]

			for i in range(0, len(ranked_candidates)):
				word = ranked_candidates[i]
				all_gold.append(gold_rankings[word])
				all_ranks.append(i)

		P = pearsonr(all_ranks, all_gold)

		return P[0]

        def evaluateRankerAccuracy(self, gold_data, rankings):
		#Binarize:
                binary_gold = []
                for line in gold_data:
                        map = {}
                        cands = line[3:]
                        for i in range(0, len(cands)-1):
                                for j in range(i+1, len(cands)):
                                        cand1 = cands[i].split(':')[1]
                                        cand2 = cands[j].split(':')[1]
                                        if cand1 not in map:
                                                map[cand1] = {}
                                        if cand2 not in map:
                                                map[cand2] = {}
                                        #Is first simpler than second?
                                        map[cand1][cand2] = 1
                                        map[cand2][cand1] = 0
                        binary_gold.append(map)

		#Calculate Accuracy:
                correct = 0.0
                total = 0.0
                for line, map in zip(rankings, binary_gold):
                        for i in range(0, len(line)-1):
                                for j in range(i+1, len(line)):
                                        cand1 = line[i]
                                        cand2 = line[j]
                                        correct += map[cand1][cand2]
                                        total += 1
                accuracy = correct/total

                return accuracy


datapath = '../../corpora/datasets/COLING_SR_Annotations_with_IDs.txt' #Path to the dataset
trainprop = float(sys.argv[1])/100.0 #Proportion of the dataset that will become training data
ntrains = int(sys.argv[2]) #Number of different models that will be trained for the same setting
step = int(sys.argv[3]) #Instance number interval
mode = sys.argv[4]

#Modes:
#- normal: regular training (no adaptitve approach is used)
#- individual: trains one for each user
#- grup-lang/age/edu/prof/all: trains one ranker for each group
#- feature-...: uses the given property as a feature

#Get feature calculator and evaluator:
fe = FeatureEstimator()
fe.addCollocationalFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/subimdb', 2, 2, 'Complexity')
fe.addWordVectorValues('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_100_cbow_retrofitted.bin', 100, 'Simplicity')
fe.addLengthFeature('Complexity')
ev = RankerEvaluator()

#Get instances and name from dataset:
instancemap, name, totalsize = mountInstances(datapath, mode)

#Produce global steps:
rawcounts = []
allsteps = []
globalpivot = int(float(totalsize)*trainprop)
for j in range(1, step, step/5)+range(step, globalpivot+1, step):
	rawcounts.append(j)
	proportion = float(j)/float(globalpivot)
	allsteps.append(proportion)


#Get train and test portions:
trainmap = {}
testmap = {}
for group in instancemap:
	pivot = int(len(instancemap[group])*trainprop)
	trainmap[group] = instancemap[group][:pivot+1]
	testmap[group] = instancemap[group][pivot+1:]

#Setup control variables:
all_results = []
for i in range(0, ntrains):
	print i
	results = []
	for stepprop in allsteps:
		preds = []
		golds = []
		for group in instancemap:
			train = trainmap[group]
			random.shuffle(train)
			test = testmap[group]
			j = max(1, int(stepprop*float(len(train))))
			used_train = train[:j]
			text_train = toText(used_train)
			ranker = OnlineRegressionRanker(fe)
			ranker.trainRegressionModel(text_train)
			ranks = ranker.getRankings(test)
			preds.extend(ranks)
			golds.extend(test)
		acc = ev.evaluateRankerAccuracy(test, ranks)
		results.append(acc)
	all_results.append(results)


#Calculate averages:
matrix = np.array(all_results)
final_scores = np.average(matrix, 0)

print len(rawcounts), len(final_scores)

#Save results:
o = open('../../corpora/adaptive/ridge_'+name+'_'+mode+'_'+str(trainprop)+'_'+str(ntrains)+'_'+str(step)+'.txt', 'w')
for i, c in zip(rawcounts, final_scores):
	o.write(str(i)+'\t'+str(c)+'\n')
o.close()
