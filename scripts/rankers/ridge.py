import sys, random
import numpy as np
from scipy.stats import *
from sklearn import linear_model
from lexenstein.features import *

def toText(data):
	result = '\n'.join(['\t'.join(line) for line in data])
	return result

def mountInstances(path):
	name = path.split('/')[-1].split('.')[0]
	f = open(path)
	insts = []
	for line in f:
		insts.append(line.strip().split('\t'))
	f.close()
	return insts, name

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

datapath = sys.argv[1] #Path to the dataset
trainprop = float(sys.argv[2])/100.0 #Proportion of the dataset that will become training data
ntrains = int(sys.argv[3]) #Number of different models that will be trained for the same setting
step = int(sys.argv[4]) #Instance number interval

#Get feature calculator and evaluator:
fe = FeatureEstimator()
fe.addCollocationalFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/subimdb', 2, 2, 'Complexity')
ev = RankerEvaluator()

#Get instances and name from dataset:
instances, name = mountInstances(datapath)

#Get train and test portions:
pivot = int(len(instances)*trainprop)
train = instances[:pivot+1]
test = instances[pivot+1:]

#Setup control variables:
all_results = []
for i in range(0, ntrains):
	random.shuffle(train)
	results = []
	for j in range(1, step)+range(step, pivot+1, step):
		used_train = train[:j]
		text_train = toText(used_train)
		ranker = OnlineRegressionRanker(fe)
		ranker.trainRegressionModel(text_train)
		ranks = ranker.getRankings(test)
		corr = ev.evaluateRanker(test, ranks)
		results.append(corr)
	all_results.append(results)

#Get list of training sizes:
sizes = range(1, step)+range(step, pivot+1, step)

#Calculate averages:
matrix = np.array(all_results)
final_scores = np.average(matrix, 0)

#Save results:
o = open('../corpora/performances/ridge_'+name+'_'+str(trainprop)+'_'+str(ntrains)+'_'+str(step)+'.txt', 'w')
for i, c in zip(sizes, final_scores):
	o.write(str(i)+'\t'+str(c)+'\n')
o.close()
