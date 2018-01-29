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
	name = 'surveymonkey'
	f = open(path)
	insts = []
	for line in f:
		data = line.strip().split('\t')
		id = data[0]
		age = int(float(data[1])/10.0)
		lang = data[2]
		langg = data[3]
		langgg = data[4]
		edu = data[5]
		prof = data[6]
		keymap = {'age':age, 'lang':lang, 'langg':langg, 'langgg':langgg, 'edu':edu, 'prof':prof}
		sent = data[7]
		target = data[8]
		sent = sent.replace('_____________', target)
		index = data[9]
		cands = set([data[10], data[11]])
		first = data[12]
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

	def trainRegressionModel(self, training_data_text, random_state):
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

		solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
		index = np.random.randint(0, len(solvers))

		self.model = linear_model.Ridge(solver=solvers[index], random_state=random_state)
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

def generateRankers(trainmap, random_state):
	rankers = {}
	for group in instancemap:
	        train = trainmap[group]
	        text_train = toText(train)
	        ranker = OnlineRegressionRanker(fe)
	        ranker.trainRegressionModel(text_train, random_state)
	        rankers[group] = ranker
	return rankers

datapath = '../../../../corpora/datasets/SurveyMonkey_150118_Data_With_IDS.txt'
trainprop = 0.5 #Proportion of the dataset that will become training data
nexp = 30 #Number of different models that will be trained for the same setting
mode = sys.argv[1]

#Modes:
#- normal: regular training (no adaptitve approach is used)
#- individual: trains one for each user
#- group-lang/langg/langgg/age/edu/prof/all: trains one ranker for each group
#- feature-...: uses the given property as a feature

#Get feature calculator and evaluator:
fe = FeatureEstimator()
fe.addCollocationalFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/subimdb', 2, 2, 'Complexity')
fe.addWordVectorValues('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_100_cbow_retrofitted.bin', 100, 'Simplicity')
fe.addLengthFeature('Complexity')
ev = RankerEvaluator()

#Get instances and name from dataset:
instancemap, name, totalsize = mountInstances(datapath, mode)

#Get train and test portions:
trainmap = {}
testmap = {}
for group in instancemap:
	pivot = int(len(instancemap[group])*trainprop)
	trainmap[group] = instancemap[group][:pivot+1]
	testmap[group] = instancemap[group][pivot+1:]

regular_results = []
rankers_list = []
for i in range(0, nexp):
	print i
	rankers = generateRankers(trainmap, i+1)
	rankers_list.append(rankers)
        preds = []
        golds = []
	for group in instancemap:
		ranker = rankers[group]
                test = testmap[group]
                ranks = ranker.getRankings(test)
                preds.extend(ranks)
                golds.extend(test)
        acc = ev.evaluateRankerAccuracy(golds, preds)
        regular_results.append(acc)

#Setup control variables:
random_results = []
for i in range(0, nexp):
	print i
	rankers = rankers_list[i]
	preds = []
	golds = []
	groupranker = instancemap.keys()
	random.shuffle(groupranker)
	grouptest = instancemap.keys()
	random.shuffle(grouptest)
	for granker, gtest in zip(groupranker, grouptest):
		ranker = rankers[granker]
		test = testmap[gtest]
		ranks = ranker.getRankings(test)
		preds.extend(ranks)
		golds.extend(test)
	acc = ev.evaluateRankerAccuracy(golds, preds)
	random_results.append(acc)

print len(random_results), np.mean(random_results), np.mean(regular_results)

#Save results:
o = open('../../../../corpora/adaptive/random_vs_proper/surveymonkey/ridge_'+name+'_'+mode+'.txt', 'w')
for reg, ran in zip(regular_results, random_results):
	o.write(str(reg)+'\t'+str(ran)+'\n')
o.close()
