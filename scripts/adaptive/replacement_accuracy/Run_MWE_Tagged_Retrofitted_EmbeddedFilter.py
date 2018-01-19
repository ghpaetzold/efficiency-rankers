from lexenstein.generators import *
from lexenstein.morphadorner import *
from lexenstein.spelling import *
import sys, os

class GlavasGenerator:

	def __init__(self, w2vmodel, pos_model, stanford_tagger, java_path):
		"""
		Creates a GlavasGenerator instance.
	
		@param w2vmodel: Binary parsed word vector model.
		For more information on how to produce the model, please refer to the LEXenstein Manual.
		"""
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = PorterStemmer()
		self.model = gensim.models.KeyedVectors.load_word2vec_format(w2vmodel, binary=True)
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus, amount):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
		#Get candidate->pos map:
		tagged_sents = self.getParsedSentences(victor_corpus)
		
		#Get initial set of substitutions:
		substitutions = self.getInitialSet(victor_corpus, tagged_sents, amount)
		return substitutions
		
	def getParsedSentences(self, victor_corpus):
		lexf = open(victor_corpus)
		sents = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			sents.append(sent)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		return tagged_sents

	def getLemmaStemMap(self, data, tsents):
		trgs = []
		trgsc = []
		trgsstems = []
		trgslemmas = []
		for i in range(0, len(data)):
			d = data[i]
			tags = tsents[i]
			target = d[1].strip().lower()
			head = int(d[2].strip())
			tag = self.getClass(tags[head][1])
			trgs.append(target)
		trgslemmas = self.lemmatizeWords(trgs)
		trgsstems = self.stemWords(trgs)
		trgmap = {}
		for i in range(0, len(trgslemmas)):
			target = data[i][1].strip().lower()
			head = int(data[i][2].strip())
			tag = self.getClass(tsents[i][head][1])
			lemma = trgslemmas[i]
			stem = trgsstems[i]
			trgmap[target] = (lemma, stem, tag)
		return trgmap
		
	def getInitialSet(self, victor_corpus, tagged_sents, amount):
		lexf = open(victor_corpus)
		data = []
		for line in lexf:
			d = line.strip().split('\t')
			data.append(d)
		lexf.close()
		
		trgmap = getLemmaStemTagMap(data, tagged_sents)
		
		subs = []

		for i in range(0, len(data)):
			d = data[i]

			word = d[1].replace(' ', '_')
			
			#If word is a phrase:
			if '_' in word:
				most_sim = []
				try:
					most_sim = self.model.most_similar(positive=[word], topn=50)
				except KeyError:
					most_sim = []

				subs.append([w[0] for w in most_sim])
			#If not:
			else:
				tword = word+'|||'+trgmap[word][2]
				print tword
				
				try:
					most_sim = self.model.most_similar(positive=[word], topn=50)
				except KeyError:
					most_sim = []
				
				subs.append([word[0] for word in most_sim if '_' not in word[0]])
			
		subs_filtered = self.filterSubs(data, subs, trgmap)
		
		final_cands = {}
		for i in range(0, len(data)):
			target = data[i][1]
			final_cands[target] = subs_filtered[i][0:min(amount, subs_filtered[i])]
		
		return final_cands
		
	def lemmatizeWords(self, words):
		result = []
		for word in words:
			result.append(self.lemmatizer.lemmatize(word))
		return result
		
	def stemWords(self, words):
		result = []
		for word in words:
			result.append(self.stemmer.stem(word))
		return result
		
	def getClass(self, tag):
		result = None
		if tag.startswith('N'):
			result = 'N'
		elif tag.startswith('V'):
			result = 'V'
		elif tag.startswith('RB'):
			result = 'A'
		elif tag.startswith('J'):
			result = 'J'
		elif tag.startswith('W'):
			result = 'W'
		elif tag.startswith('PRP'):
			result = 'P'
		else:
			result = tag.strip()
		return result
	
	def filterSubs(self, data, subs, trgmap):
		result = []

		prohibited_edges = set([line.strip() for line in open('/export/data/ghpaetzold/benchmarking/phrase_ls/corpora/prohibited_edges.txt')])
		prohibited_chars = set([line.strip() for line in open('/export/data/ghpaetzold/benchmarking/phrase_ls/corpora/prohibited_chars.txt')])
		vowels = set('aeiouyw')
		consonants = set('bcdfghjklmnpqrstvxz')

		for i in range(0, len(data)):
			d = data[i]
			
			sent = d[0].split(' ')
			index = int(d[2])
			if index==0:
				prevtgt = 'NULL'
			else:
				prevtgt = sent[index-1]
			if index==len(sent)-1:
				proxtgt = 'NULL'
			else:
				proxtgt = sent[index+1]

			target = d[1]
			targett = target.split(' ')
			firsttgt = targett[0]
		    lasttgt = targett[-1]
			
			if len(targett)>1:
				most_sim = subs[i]
				most_simf = []

				for cand in most_sim:
					c = cand.replace('_', ' ')
					if '|||' in c:
						c = c.split('|||')[0]
					tokens = c.split(' ')
					first = tokens[0]
					last = tokens[-1]
					cchars = set(c)
					edges = set([first, last])
					inter_edge = edges.intersection(prohibited_edges)
								inter_chars = cchars.intersection(prohibited_chars)
					if c not in target and target not in c and first!=prevtgt and last!=proxtgt:
						if len(inter_edge)==0 and len(inter_chars)==0:
							if (firsttgt=='most' and first!='more') or (firsttgt=='more' and first!='most') or (firsttgt!='more' and firsttgt!='most'):
								if (prevtgt=='an' and c[0] in vowels) or (prevtgt=='a' and c[0] in consonants) or (prevtgt!='an' and prevtgt!='a'):
									most_simf.append(c)

				result.append(most_simf)
			else:
				tstem = trgmap[target][0]
				tlemma = trgmap[target][1]
				tag = trgmap[target][2]

				rtarget = t+'|||'+tag

				most_sim = subs[i]
				most_simf = []

				for cand in most_sim:
					candd = cand.split('|||')
					cword = candd[0].strip()
					ctag = candd[1].strip()

					if ctag==tag:
						if (not tlemma in cword) and (not tstem in cword) and (cword not in target) and (target not in cword):
							most_simf.append(cand)
			
				result.append(most_simf)
		return result










victor_corpus = sys.argv[1].strip()

w2v = '/export/data/ghpaetzold/word2vecvectors/models/word_vectors_mweall_generalized_1300_cbow_retrofitted.bin'

kg = GlavasGenerator(w2v)
subs = kg.getSubstitutions(victor_corpus, 40)

os.system('mkdir ../../../corpora/substitutions/')
for i in [5, 10, 20, 30, 40]:
	out = open('../../../corpora/substitutions/mweretrofittedpaetzoldfembed_substitutions_'+str(i)+'.txt', 'w')
	for k in subs.keys():
		newline = k + '\t'
		if len(subs[k])>0:
			for c in subs[k][:i]:
				newline += c + '|||'
			newline = newline[0:len(newline)-3]
			out.write(newline.strip() + '\n')
	out.close()
		
