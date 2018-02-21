import sys, random
import numpy as np
from scipy.stats import *
from sklearn import linear_model
from lexenstein.features import *
from sklearn.cross_validation import train_test_split

def getIndex(sent, target):
        tokens = sent.split(' ')
        for i in range(len(tokens)):
                if tokens[i]==target:
                        return i
        return None

def mountGoldInstances(path):
        insts = []
        f = open(path)
        sent = f.readline().lower().strip()
        while len(sent)>0:
                inst = [sent]
                target = f.readline().lower().strip()
                inst.append(target)
                inst.append(getIndex(sent, target))
                f.readline()
                i = 0
                cand = f.readline().strip().lower()
                while len(cand)>0:
                        i += 1
                        inst.append(str(i)+':'+cand)
                        cand = f.readline().strip().lower()
                sent = f.readline().lower().strip()
                insts.append(inst)
        f.close()
        return insts

def toText(data):
	result = '\n'.join(['\t'.join(line) for line in data])
	return result

def mountTestInstances(candfile):
        f = open(candfile)
        insts = []
        for line in f:
                insts.append(line.strip().split('\t'))
        f.close()
        return insts

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

class BoundaryRanker:

	def __init__(self, fe):
		self.fe = fe
		self.classifier = None
		self.feature_selector = None

	def trainRanker(self, training_data_text, positive_range, loss, penalty, alpha, l1_ratio, epsilon):
                data = [d.split('\t') for d in training_data_text.split('\n')]

                #Create matrixes:
                X = self.fe.calculateFeatures(training_data_text, input='text')
                Y = self.generateLabels(data, positive_range)

                #Train classifier:
                self.classifier = linear_model.SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, epsilon=epsilon)
		self.classifier.fit(X, Y)
		
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

	def evaluateReplacementAccuracy(self, gold_data, rankings):
                acc = 0.0
                total = 0.0
                for ginst, rinst in zip(gold_data, rankings):
                        if len(rinst)==0:
                                total += 1.0
                        else:
                                golds = set([w.strip().split(':')[1] for w in ginst[3:]])
                                first = rinst[0]
                                if first in golds:
                                        acc += 1.0
                                total += 1.0
                return acc/total

	def evaluateReplacementSimplicity(self, gold_data, rankings):
                simp = 0.0
                total = 0.0
                for ginst, rinst in zip(gold_data, rankings):
                        if len(rinst)>0:
                                gold_map = {}
                                for cand in ginst[3:]:
                                        canddata = cand.split(':')
                                        rank = int(canddata[0])
                                        cand = canddata[1]
                                        gold_map[cand] = rank
                                first = rinst[0]
                                if first in gold_map:
                                        simp += gold_map[first]
                                        total += 1.0
                if total==0:
                        res = 0.0
                else:
                        res = simp/total
                return res

datapath = '../../../../corpora/datasets/SurveyMonkey_150118_Data_With_IDS.txt'
candfile = '../../../../corpora/substitutions/mweretrofittedpaetzoldfembed_substitutions_victor_'+str(sys.argv[1])+'.txt'
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
#fe.addWordVectorValues('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_100_cbow_retrofitted.bin', 100, 'Simplicity')
fe.addLengthFeature('Complexity')
ev = RankerEvaluator()

#Get instances and name from dataset:
trainmap, name, totalsize = mountInstances(datapath, mode)
testinsts = mountTestInstances(candfile)
goldinsts = mountGoldInstances('../../../../corpora/datasets/adjusted_selected_instances_tok.txt')

#Produce global steps:
rawcounts = []
allsteps = []
globalpivot = int(totalsize)
for j in range(1, step, step/5)+range(step, globalpivot+1, step):
        rawcounts.append(j)
        proportion = float(j)/float(globalpivot)
        allsteps.append(proportion)

#Setup control variables:
all_results = []
for i in range(0, ntrains):
	print 'Iteration: ', i
        results = []
        for k, stepprop in enumerate(allsteps):
                preds = []
                golds = []
                for group in trainmap:
                        train = trainmap[group]
                        test = testinsts
                        j = max(1, int(stepprop*float(len(train))))
                        used_train = train[:j]
                        text_train = toText(used_train)
                        ranker = BoundaryRanker(fe)
#                        ranker.trainRankerWithCrossValidation(text_train, 1, 2, 0.5)
			ranker.trainRanker(text_train, 1, 'hinge', 'elasticnet', 0.001, 0.0, 0.001)
                        ranks = ranker.getRankings(test)
                        preds.extend(ranks)
                        golds.extend(goldinsts)
                simp = ev.evaluateReplacementSimplicity(golds, preds)
                results.append(simp)
        all_results.append(results)

#Calculate averages:
matrix = np.array(all_results)
final_scores = np.average(matrix, 0)

print len(rawcounts), len(final_scores)

#Save results:
o = open('../../../../corpora/adaptive/replacement_simplicity/surveymonkey/boundary_'+name+'_'+mode+'_'+str(sys.argv[1])+'_'+str(ntrains)+'_'+str(step)+'.txt', 'w')
for i, c in zip(rawcounts, final_scores):
	o.write(str(i)+'\t'+str(c)+'\n')
o.close()
