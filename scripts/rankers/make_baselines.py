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

class MetricRanker:

	def __init__(self, fe):
		self.fe = fe
		self.feature_values = None
		
	def getRankings(self, data, featureIndex):
		#Transform data:
		textdata = ''
		for inst in data:
			for token in inst:
				textdata += token+'\t'
			textdata += '\n'
		textdata = textdata.strip()
		#If feature values are not available, then estimate them:
		self.feature_values = self.fe.calculateFeatures(textdata, input='text')
		
		#Create object for results:
		result = []
		
		#Read feature values for each candidate in victor corpus:
		index = 0
		for line in data:
			substitutions = line[3:]
			
			#Create dictionary of substitution to feature value:
			scores = {}
			for substitution in substitutions:
				word = substitution.strip().split(':')[1].strip()
				scores[word] = self.feature_values[index][featureIndex]
				index += 1
			
			#Check if feature is simplicity or complexity measure:
			rev = False
			if self.fe.identifiers[featureIndex][1]=='Simplicity':
				rev = True
			
			#Sort substitutions:
			sorted_substitutions = sorted(scores.keys(), key=scores.__getitem__, reverse=rev)
		
			#Add them to result:
			result.append(sorted_substitutions)
		
		#Return result:
		return result
		
	def size(self):
		return len(self.fe.identifiers)

class GlavasRanker:

	def __init__(self, fe):
		
		self.fe = fe
		self.feature_values = None
		
	def getRankings(self, data):	
		#Transform data:
		textdata = ''
		for inst in data:
			for token in inst:
				textdata += token+'\t'
			textdata += '\n'
		textdata = textdata.strip()
		
		#If feature values are not available, then estimate them:
		self.feature_values = self.fe.calculateFeatures(textdata, input='text')
		
		#Create object for results:
		result = []
		
		#Read feature values for each candidate in victor corpus:
		index = 0
		for line in data:
			#Get all substitutions in ranking instance:
			substitutions = line[3:]
			
			#Get instance's feature values:
			instance_features = []
			for substitution in substitutions:
				instance_features.append(self.feature_values[index])
				index += 1
			
			rankings = {}
			for i in range(0, len(self.fe.identifiers)):
				#Create dictionary of substitution to feature value:
				scores = {}
				for j in range(0, len(substitutions)):
					substitution = substitutions[j]
					word = substitution.strip().split(':')[1].strip()
					scores[word] = instance_features[j][i]
				
				#Check if feature is simplicity or complexity measure:
				rev = False
				if self.fe.identifiers[i][1]=='Simplicity':
					rev = True
				
				#Sort substitutions:
				words = scores.keys()
				sorted_substitutions = sorted(words, key=scores.__getitem__, reverse=rev)
				
				#Update rankings:
				for j in range(0, len(sorted_substitutions)):
					word = sorted_substitutions[j]
					if word in rankings:
						rankings[word] += j
					else:
						rankings[word] = j
		
			#Produce final rankings:
			final_rankings = sorted(rankings.keys(), key=rankings.__getitem__)
		
			#Add them to result:
			result.append(final_rankings)
		
		#Return result:
		return result

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
		S, p = spearmanr(all_ranks, all_gold)
		P = pearsonr(all_ranks, all_gold)
		return S#P[0]

datapath = sys.argv[1] #Path to the dataset

#Get feature calculator and evaluator:
fe = FeatureEstimator()
#fe.addNGramProbabilityFeature('/export/data/ghpaetzold/subtitlesimdb/corpora/160715/subtleximdb.5gram.unk.bin.txt', 1, 0, 'Simplicity')
#fe.addNGramProbabilityFeature('/export/data/ghpaetzold/subtitlesimdb/corpora/160715/subtleximdb.5gram.unk.bin.txt', 0, 1, 'Simplicity')
#fe.addNGramProbabilityFeature('/export/data/ghpaetzold/subtitlesimdb/corpora/160715/subtleximdb.5gram.unk.bin.txt', 1, 1, 'Simplicity')
#fe.addNGramProbabilityFeature('/export/data/ghpaetzold/subtitlesimdb/corpora/160715/subtleximdb.5gram.unk.bin.txt', 2, 0, 'Simplicity')
#fe.addNGramProbabilityFeature('/export/data/ghpaetzold/subtitlesimdb/corpora/160715/subtleximdb.5gram.unk.bin.txt', 0, 2, 'Simplicity')
#fe.addCollocationalFeature('/export/data/ghpaetzold/subtitlesimdb/corpora/160715/subtleximdb.5gram.unk.bin.txt', 1, 1, 'Simplicity')
#w2vmodel = '/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_200_glove.bin'
#fe.addWordVectorSimilarityFeature(w2vmodel, 'Simplicity')
#fe.addWordVectorContextSimilarityFeature(w2vmodel, model, tagger, java, 'Simplicity')
fe.addCollocationalFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/subimdb', 2, 2, 'Simplicity')
ev = RankerEvaluator()

instances, name = mountInstances(datapath)

gr = GlavasRanker(fe)
mr = MetricRanker(fe)

#Setup control variables:
corrs_g = []
corrs_f = []
for prop in range(20, 120, 20):
	trainprop = float(prop)/100.0
	pivot = int(len(instances)*trainprop)
        random.shuffle(instances)
	test = instances[:pivot]
	print len(test)
	ranks_g = gr.getRankings(test)
	ranks_f = mr.getRankings(test, 0)
        corr_g = ev.evaluateRanker(test, ranks_g)
	corr_f = ev.evaluateRanker(test, ranks_f)
        corrs_g.append(corr_g)
	corrs_f.append(corr_f)

print corrs_g
print corrs_f

#Save results:
o = open('../../corpora/baselines/'+name+'_glavas.txt', 'w')
o.write(str(numpy.mean(corrs_g))+'\n')
o.close()

o = open('../../corpora/baselines/'+name+'_frequency.txt', 'w')
o.write(str(numpy.mean(corrs_f))+'\n')
o.close()
