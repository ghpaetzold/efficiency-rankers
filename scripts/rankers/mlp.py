import sys, random
import numpy as np
from scipy.stats import *
from sklearn import linear_model
from lexenstein.features import *
from keras.optimizers import *
from keras.models import *
from keras.layers.core import *

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

class NNRegressionRanker:

	def __init__(self, fe, model=None):
		self.fe = fe
		self.model = model
		
	def createRanker(self, layers, hidden_size):
		model = Sequential()
		model.add(Dense(units=hidden_size, input_dim=len(self.fe.identifiers)*2, kernel_initializer="glorot_uniform"))
		model.add(Activation("tanh"))
		model.add(Dropout(0.25))
		for i in range(0, layers):
			model.add(Dense(units=hidden_size, kernel_initializer="glorot_uniform"))
			model.add(Activation("tanh"))
			model.add(Dropout(0.10))
		model.add(Dense(units=1))
		model.add(Activation("linear"))
		model.compile(loss='mean_squared_error', optimizer='adam')
		self.model = model
		return model
		
	def saveRanker(self, json_path, h5_path):
		json_string = self.model.to_json()
		open(json_path, 'w').write(json_string)
		self.model.save_weights(h5_path, overwrite=True)
		
	def loadRanker(self, json_path, h5_path):
		model = model_from_json(open(json_path).read())
		model.load_weights(h5_path)
		model.compile(loss='mean_squared_error', optimizer='adam')
		self.model = model
		return model
		
	def trainRanker(self, training_data_text, epochs, batch_size):
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
		self.model.fit(Xtr, Ytr, nb_epoch=epochs, batch_size=batch_size, verbose=0)
		
	def getRankings(self, data):
		#Transform data:
		textdata = ''
		for inst in data:
			for token in inst:
				textdata += token+'\t'
			textdata += '\n'
		textdata = textdata.strip()
		features = self.fe.calculateFeatures(textdata, input='text')
		ranks = []
		c = -1
		index = 0
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
		ranker = NNRegressionRanker(fe)
		ranker.createRanker(4, 8)
		ranker.trainRanker(text_train, 200, 1000)
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
o = open('../../corpora/performances/mlp_'+name+'_'+str(trainprop)+'_'+str(ntrains)+'_'+str(step)+'.txt', 'w')
for i, c in zip(sizes, final_scores):
	o.write(str(i)+'\t'+str(c)+'\n')
o.close()
