import sys, random, codecs
import numpy as np
from scipy.stats import *
from sklearn import linear_model
from lexenstein.features import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.optimizers import *
from keras.models import *
from keras.layers.core import *
from keras.layers import *
from keras.models import load_model
from keras.callbacks import *
from sklearn.metrics import log_loss

def getOneHotMatrix(data, profilemaps):
	sizes = []
	sizes.append(len(profilemaps['age'].keys()))
	sizes.append(len(profilemaps['lang'].keys()))
	sizes.append(len(profilemaps['langg'].keys()))
	sizes.append(len(profilemaps['langgg'].keys()))
	sizes.append(len(profilemaps['edu'].keys()))
	sizes.append(len(profilemaps['prof'].keys()))

	vars = ['age', 'lang', 'langg', 'langgg', 'edu', 'prof']

	results = []
	for line in data:
		onehots = []
		for i, var in enumerate(line):
			vec = np.zeros(sizes[i])
			vec[var] = 1.0
			onehots.append(vec)
		results.append(np.concatenate(onehots))

	return np.array(results)

def getProfileMaps(path):
	map = {}
	files = os.listdir(path)
	for file in files:
		name = file[:-4]
		vmap = {}
		f = codecs.open(path+'/'+file, encoding='utf8')
		for line in f:
			data = line.strip().split('\t')
			v = data[0].strip()
			l = int(data[1].strip())
			vmap[v] = l
		map[name] = vmap
		f.close()
	return map

def toText(data):
	result = '\n'.join(['\t'.join(line) for line in data])
	return result

def mountInstances(path, mode, profilemaps, fe, random_words):
	allinsts = []
	allproffeats = []
	artificiality = []
	size = 0
	name = 'surveymonkey'
	f = open(path)
	inst_to_cands = {}
	ss_labels = []

	for line in f:
		data = line.strip().split('\t')
		sent = data[7]
		target = data[8]
		sent = sent.replace('_____________', target)
		index = data[9]
		cands = set([data[10], data[11]])
		instdata = (sent, target, index)
		if instdata not in inst_to_cands:
			inst_to_cands[instdata] = set([])
		inst_to_cands[instdata].update(cands)
	f.close()

	f = open(path)
	profiles = {}
	for line in f:
		data = line.strip().split('\t')
		id = data[0]
		age = int(float(data[1])/10.0)*10
		if age>=70:
			age = '70'
		else:
			age = str(age)
		lang = data[2].strip()
		langg = data[3].strip()
		langgg = data[4].strip()
		edu = data[5].strip()
		prof = data[6].strip()
		keymap = {'age':age, 'lang':lang, 'langg':langg, 'langgg':langgg, 'edu':edu, 'prof':prof}
		sent = data[7]
		target = data[8]
		sent = sent.replace('_____________', target)
		index = data[9]
		cands = set([data[10], data[11]])
		first = data[12]
		cands.remove(first)
		second = list(cands)[0]
		profile = (age, lang, langg, langgg, edu, prof)

		featv = [profilemaps['age'][age], profilemaps['lang'][lang], profilemaps['langg'][langg], profilemaps['langgg'][langgg], profilemaps['edu'][edu], profilemaps['prof'][prof]]
		allproffeats.append(featv)
		ss_labels.append([1.0, 1.0])

		inst = [sent, target, index, '1:'+first, '2:'+second]
		allinsts.append(inst)
		artificiality.append(False)

		if profile not in profiles:
			profiles[profile] = set([])
		profiles[profile] = featv

		size += 1
	f.close()

	for profile in profiles:
		featv = profiles[profile]
		for inst in inst_to_cands:
			cands = inst_to_cands[inst]
			for cand in cands:
				rand_word = random_words[np.random.randint(0, len(random_words)-1)]
				allproffeats.append(featv)
				allinsts.append([inst[0], inst[1], inst[2], '1:'+cand, '2:'+rand_word])
				artificiality.append(True)
				ss_labels.append([1.0, 0.0])

	#Transform profile features in one hot representation:
	allproffeats = getOneHotMatrix(allproffeats, profilemaps)

	#Create a permutation to shuffle instances to balance ss labels:
	permut = np.random.permutation(len(allinsts))
	newallinsts = []
	newartificiality = []
	for index in permut:
		newallinsts.append(allinsts[index])
		newartificiality.append(artificiality[index])
	allinsts = newallinsts
	artificiality = newartificiality

	#Adjust profile feature vectors according to permutation, duplicating them to match number of textual features:
	rows = len(allproffeats)
	columns = len(allproffeats[0])
	allproffeats = allproffeats[permut].reshape(rows, 1, columns)
	allproffeats = np.concatenate((allproffeats, allproffeats), axis=1).reshape(2*rows, columns)

	#Adjust ss labels in the same way:
	rows = len(ss_labels)
        columns = len(ss_labels[0])
	ss_labels = np.array(ss_labels)[permut].reshape(rows, 1, columns)
	ss_labels = np.concatenate((ss_labels, ss_labels), axis=1).reshape(2*rows, columns)

	alltextinsts = toText(allinsts)
	
	alltextfeats = np.array(fe.calculateFeatures(alltextinsts, format='victor', input='text'))

	nfeatures = len(alltextfeats[0])*2+len(allproffeats[0])

	return allinsts, allproffeats, alltextfeats, name, size, nfeatures, ss_labels, artificiality

class HybridNeuralRegressionRanker:

	def __init__(self, fe, n_features, layers=2, layer_size=50):
		self.fe = fe
		input = Input(shape=(n_features,), name='input')
		dense = Dense(layer_size, activation='relu')(input)
		for i in range(0, layers-1):
			dense = Dense(layer_size, activation='relu')(dense)
		ss_out = Dense(2, activation='sigmoid', name='ss_out')(dense)
		sr_out = Dense(1, activation='linear', name='sr_out')(dense)
		self.model = Model(input=[input], output=[ss_out, sr_out])
		self.model.compile(optimizer='rmsprop', loss={'ss_out': 'binary_crossentropy', 'sr_out': 'mean_absolute_error'}, loss_weights={'ss_out': 1.0, 'sr_out': 1.0})

	def trainRegressionModel(self, training_data_text, text_features, profile_features, ss_labels):
		Xtr = []
		Ytr_ss = []
		Ytr_sr = []
		c = -1
		for line in training_data_text.strip().split('\n'):
			data = line.strip().split('\t')
			cands = [cand.strip().split(':')[1] for cand in data[3:]]
			indexes = [int(cand.strip().split(':')[0]) for cand in data[3:]]
			infomap = {}
			for cand in cands:
				c += 1
				infomap[cand] = (text_features[c], profile_features[c], ss_labels[c])
			for i in range(0, len(cands)-1):
				for j in range(i+1, len(cands)):
					indexi = indexes[i]
					indexj = indexes[j]
					indexdiffji = indexj-indexi
					indexdiffij = indexi-indexj
					positive = infomap[cands[i]][0]
					negative = infomap[cands[j]][0]
					profilefeats = infomap[cands[i]][1]
					sslabel1 = infomap[cands[i]][2]
					sslabel2 = sslabel1
					if sslabel2[1]==0.0:
						sslabel2[1] = 1.0
						sslabel2[0] = 0.0
					v1 = np.concatenate((positive,negative,profilefeats))
					v2 = np.concatenate((negative,positive,profilefeats))
					Xtr.append(v1)
					Xtr.append(v2)
					Ytr_ss.append(sslabel1)
					Ytr_ss.append(sslabel2)
					Ytr_sr.append(indexdiffji)
					Ytr_sr.append(indexdiffij)
		Xtr = np.array(Xtr)
		sslabels = np.array(Ytr_ss)
		srlabels = np.array(Ytr_sr)

		callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

		self.model.fit(Xtr, {'ss_out': sslabels, 'sr_out': srlabels}, epochs=75, batch_size=32, verbose=0, validation_split=0.25, callbacks=[callback])
		return self.model

	def getRankings(self, data, text_features, profile_features):
		ss_probs = []
		sr_ranks = []
		c = -1
		for line in data:
			cands = [cand.strip().split(':')[1].strip() for cand in line[3:]]
			infomap = {}
			scoremap = {}
			for cand in cands:
				c += 1
				infomap[cand] = (text_features[c], profile_features[c])
				scoremap[cand] = 0.0
			for i in range(0, len(cands)-1):
				cand1 = cands[i]
				for j in range(i+1, len(cands)):
					cand2 = cands[j]
					posneg = np.concatenate((infomap[cand1][0], infomap[cand2][0], infomap[cand1][1]))
					probs = self.model.predict(np.array([posneg]))
					ss_predictions = probs[0][0]
					for value in ss_predictions:
						ss_probs.append(value)
					sr_prediction =  probs[1][0]
					scoremap[cand1] += sr_prediction
					#negpos = np.concatenate((featmap[cand2], featmap[cand1]))
					#probs = self.model.predict(np.array([negpos]))
					#score = probs[0]
					#scoremap[cand1] -= score
			rank = sorted(scoremap.keys(), key=scoremap.__getitem__, reverse=True)
			if len(rank)>1:
				if rank[0]==line[1].strip():
					rank = rank[1:]
			sr_ranks.append(rank)
		return ss_probs, sr_ranks

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

        def evaluateRankerAccuracy(self, gold_data, selections, rankings, ss_labels, artif):
		ss_gold = []
		for i in range(0, len(ss_labels), 2):
			values = ss_labels[i]
			ss_gold.append(values[0])
			ss_gold.append(values[1])
		ss_pred = []
		for selection in selections:
			if selection>0.5:
				ss_pred.append(1.0)
			else:
				ss_pred.append(0.0)

		ss_correct = 0.0
		for ssgold, sspred in zip(ss_gold, ss_pred):
			if ssgold==sspred:
				ss_correct += 1.0
		ss_accuracy = ss_correct/float(len(ss_gold))

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
                for line, map, art in zip(rankings, binary_gold, artif):
			if not art:
	                        for i in range(0, len(line)-1):
	                                for j in range(i+1, len(line)):
	                                        cand1 = line[i]
	                                        cand2 = line[j]
	                                        correct += map[cand1][cand2]
	                                        total += 1
                sr_accuracy = correct/total

                return ss_accuracy, sr_accuracy

datapath = '../../../../corpora/datasets/SurveyMonkey_150118_Data_With_IDS.txt'
trainprop = float(sys.argv[1])/100.0 #Proportion of the dataset that will become training data
ntrains = int(sys.argv[2]) #Number of different models that will be trained for the same setting
step = int(sys.argv[3]) #Instance number interval
layers = int(sys.argv[4])
layer_size = int(sys.argv[5])

#Get useful list of words:
random_words = set([])
f = open('/export/data/ghpaetzold/corpora/simplewiki.txt')
for line in f:
	random_words.update(line.strip().lower().split(' '))
f.close()
random_words = list(random_words)

#Get feature calculator and evaluator:
fe = FeatureEstimator()
fe.addCollocationalFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/subimdb', 2, 2, 'Complexity')
fe.addWordVectorValues('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_300_cbow_retrofitted.bin', 300, 'Simplicity')
fe.addSentenceProbabilityFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/subimdb', 'Simplicity')
fe.addSentenceProbabilityFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/simplewiki', 'Simplicity')
fe.addSentenceProbabilityFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/wikipedia', 'Simplicity')
fe.addLengthFeature('Simplicity')
fe.addNumberOfTokensFeature('Simplicity')
fe.addMinimumTokenProbabilityFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/simplewiki', 'Simplicity')
fe.addMaximumTokenProbabilityFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/simplewiki', 'Simplicity')
fe.addAverageTokenProbabilityFeature('/export/data/ghpaetzold/subimdbexperiments/corpora/binlms/simplewiki', 'Simplicity')
fe.addMinimumWordVectorSimilarityFeature('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_300_cbow_retrofitted.bin', 'Simplicity')
fe.addMaximumWordVectorSimilarityFeature('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_300_cbow_retrofitted.bin', 'Simplicity')
fe.addAverageWordVectorSimilarityFeature('/export/data/ghpaetzold/word2vecvectors/models/word_vectors_all_300_cbow_retrofitted.bin', 'Simplicity')

ev = RankerEvaluator()

#Get profile feature maps:
profilemaps = getProfileMaps('../../../../corpora/datasets/maps/')

#Get instances and name from dataset:
instances, proffeats, textfeats, name, totalsize, nfeatures, ss_labels, artificiality = mountInstances(datapath, mode, profilemaps, fe, random_words)

#Produce global steps:
rawcounts = []
allsteps = []
globalpivot = int(float(totalsize)*trainprop)
for j in range(1, step, step/5)+range(step, globalpivot+1, step):
	rawcounts.append(j)
	proportion = float(j)/float(globalpivot)
	allsteps.append(proportion)

#Get train and test portions:
pivot = int(len(instances)*trainprop)
traininsts = instances[:pivot+1]
trainartif = artificiality[:pivot+1]
pfeatstrain = proffeats[:(pivot+1)*2]
tfeatstrain = textfeats[:(pivot+1)*2]
sslabelstrain = ss_labels[:(pivot+1)*2]

testinsts = instances[pivot+1:]
testartif = artificiality[pivot+1:]
pfeatstest = proffeats[(pivot+1)*2:]
tfeatstest = textfeats[(pivot+1)*2:]
sslabelstest = ss_labels[(pivot+1)*2:]

#Setup control variables:
ss_all_results = []
sr_all_results = []
for i in range(0, ntrains):
	print 'Iteration: ', i
	ss_results = []
	sr_results = []
	for stepprop in allsteps:
		print '\tStep: ', stepprop
		j = max(1, int(stepprop*float(len(traininsts))))
		used_train = traininsts[:j]
		used_train_text_feats = tfeatstrain[:2*j]
		used_train_prof_feats = pfeatstrain[:2*j]
		used_train_ss_labels = sslabelstrain[:2*j]
		text_train = toText(used_train)
		ranker = HybridNeuralRegressionRanker(fe, nfeatures, layers=layers, layer_size=layer_size)
		ranker.trainRegressionModel(text_train, used_train_text_feats, used_train_prof_feats, used_train_ss_labels)
		ss_probs, sr_ranks = ranker.getRankings(testinsts, tfeatstest, pfeatstest)
		ss_acc, sr_acc = ev.evaluateRankerAccuracy(testinsts, ss_probs, sr_ranks, sslabelstest, testartif)
		print '\t Performance: ', i, stepprop, ss_acc, sr_acc
		ss_results.append(ss_acc)
		sr_results.append(sr_acc)
	ss_all_results.append(ss_results)
	sr_all_results.append(sr_results)


#Calculate averages:
ss_matrix = np.array(ss_all_results)
ss_final_scores = np.average(ss_matrix, 0)
sr_matrix = np.array(sr_all_results)
sr_final_scores = np.average(sr_matrix, 0)

print len(rawcounts), len(sr_final_scores)

#Save results:
o = open('../../../../corpora/adaptive/hybrid_neural/surveymonkey/hybridneuralwithfeats_'+name+'_'+str(trainprop)+'_'+str(ntrains)+'_'+str(step)+'_'+str(layers)+'_'+str(layer_size)+'.txt', 'w')
for i, c1, c2 in zip(rawcounts, ss_final_scores, sr_final_scores):
	o.write(str(i)+'\t'+str(c1)+'\t'+str(c2)+'\n')
o.close()
