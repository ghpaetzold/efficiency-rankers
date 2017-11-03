import sys, random
import numpy as np
from scipy.stats import *
from sklearn import linear_model
from lexenstein.features import *
from sklearn.cross_validation import train_test_split

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

class BoundaryRanker:

	def __init__(self, fe):
		self.fe = fe
		self.classifier = None
		self.feature_selector = None
		
	def trainRankerWithCrossValidation(self, training_data_text, positive_range, folds, test_size, losses=['hinge', 'modified_huber'], penalties=['elasticnet'], alphas=[0.0001, 0.001, 0.01], l1_ratios=[0.0, 0.15, 0.25, 0.5, 0.75, 1.0], k='all'):
		#Read victor corpus:
		data = [d.split('\t') for d in training_data_text.split('\n')]
		
		#Create matrixes:
		X = self.fe.calculateFeatures(training_data_text, input='text')
		Y = self.generateLabels(data, positive_range)

		#Extract ranking problems:
		firsts = []
		candidates = []
		Xsets = []
		Ysets = []
		index = -1
		for line in data:
			fs = set([])
			cs = []
			Xs = []
			Ys = []
			for cand in line[3:len(line)]:
				index += 1
				candd = cand.split(':')
				rank = candd[0].strip()
				word = candd[1].strip()
				
				cs.append(word)
				Xs.append(X[index])
				Ys.append(Y[index])
				if rank=='1':
					fs.add(word)
			firsts.append(fs)
			candidates.append(cs)
			Xsets.append(Xs)
			Ysets.append(Ys)
		
		#Create data splits:
		datasets = []
		for i in range(0, folds):
			Xtr, Xte, Ytr, Yte, Ftr, Fte, Ctr, Cte = train_test_split(Xsets, Ysets, firsts, candidates, test_size=test_size, random_state=i)
			Xtra = []
			for matrix in Xtr:
				Xtra += matrix
			Xtea = []
			for matrix in Xte:
				Xtea += matrix
			Ytra = []
			for matrix in Ytr:
				Ytra += matrix
			datasets.append((Xtra, Ytra, Xte, Xtea, Fte, Cte))
		
		#Get classifier with best parameters:
		max_score = -1.0
		parameters = ()
		for l in losses:
			for p in penalties:
				for a in alphas:
					for r in l1_ratios:
						sum = 0.0
						sum_total = 0
						for dataset in datasets:
							Xtra = dataset[0]
							Ytra = dataset[1]
							Xte = dataset[2]
							Xtea = dataset[3]
							Fte = dataset[4]
							Cte = dataset[5]

							classifier = linear_model.SGDClassifier(loss=l, penalty=p, alpha=a, l1_ratio=r, epsilon=0.0001)
							try:
								classifier.fit(Xtra, Ytra)
								t1 = self.getCrossValidationScore(classifier, Xtea, Xte, Fte, Cte)
								sum += t1
								sum_total += 1
							except Exception:
								pass
						sum_total = max(1, sum_total)
						if (sum/sum_total)>max_score:
							max_score = sum
							parameters = (l, p, a, r)
		self.classifier = linear_model.SGDClassifier(loss=parameters[0], penalty=parameters[1], alpha=parameters[2], l1_ratio=parameters[3], epsilon=0.0001)
		self.classifier.fit(X, Y)
	
	def getCrossValidationScore(self, classifier, Xtea, Xte, firsts, candidates):
		distances = classifier.decision_function(Xtea)
		index = -1
		corrects = 0
		total = 0
		for i in range(0, len(Xte)):
			xset = Xte[i]
			maxd = -999999
			for j in range(0, len(xset)):
				index += 1
				distance = distances[index]
				if distance>maxd:
					maxd = distance
					maxc = candidates[i][j]
			if maxc in firsts[i]:
				corrects += 1
			total += 1
		return float(corrects)/float(total)
	
	def getRankings(self, data):
		#Transform data:
                textdata = ''
                for inst in data:
                        for token in inst:
                                textdata += token+'\t'
                        textdata += '\n'
                textdata = textdata.strip()
		
		#Create matrixes:
		X = self.fe.calculateFeatures(textdata, input='text')
		
		#Get boundary distances:
		distances = self.classifier.decision_function(X)
		
		#Get rankings:
		result = []
		index = 0
		for i in range(0, len(data)):
			line = data[i]
			scores = {}
			for subst in line[3:len(line)]:
				word = subst.strip().split(':')[1].strip()
				scores[word] = distances[index]
				index += 1
			ranking_data = sorted(scores.keys(), key=scores.__getitem__, reverse=True)
			result.append(ranking_data)
		
		#Return rankings:
		return result

	def generateLabels(self, data, positive_range):
		Y = []
		for line in data:
			last_rank = int(line[len(line)-1].split(':')[0].strip())
			max_range = min(last_rank, positive_range)
			if last_rank<=positive_range:
				max_range -= 1
			for i in range(3, len(line)):
				rank_index = int(line[i].split(':')[0].strip())
				if rank_index<=max_range:
					Y.append(1)
				else:
					Y.append(0)
		print set(Y)
		return Y


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
		ranker = BoundaryRanker(fe)
		ranker.trainRankerWithCrossValidation(text_train, 3, 2, 0.5)
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
o = open('../corpora/performances/boundary_'+name+'_'+str(trainprop)+'_'+str(ntrains)+'_'+str(step)+'.txt', 'w')
for i, c in zip(sizes, final_scores):
	o.write(str(i)+'\t'+str(c)+'\n')
o.close()
