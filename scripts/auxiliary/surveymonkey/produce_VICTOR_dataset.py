def getIndex(target, sent):
	newtgt = target.lower().replace(' ', '_')
	newsent = sent.lower().replace(target.lower(), newtgt)
	tokens = newsent.split(' ')
	for i, token in enumerate(tokens):
		if newtgt==token:
			return i
	return -1
	
def getUnderscoredSent(sent, target, tgtindex):
	return sent.lower().replace(target, target.replace(' ', '_'))

#Get instances:
f = open('C:/Users/ghpae/Google Drive/User Studies/Adaptive_Ranking/Corpora/SCC instances/adjusted_selected_instances.txt')
f2 = open('C:/Users/ghpae/Google Drive/User Studies/Adaptive_Ranking/Corpora/SCC instances/adjusted_selected_instances_tok.txt')
truemap = {}
sent = f.readline().strip()
sentok = f2.readline().strip()
count = 0
truesents = []
truetargets = []
trueindexes = []
truecands = []
while len(sentok)>0:
	count += 1
	target = f.readline().strip()
	f2.readline()
	
	truetargets.append(target)
	tgtindex = getIndex(target, sentok)
	trueindexes.append(tgtindex)
	fixedsent = getUnderscoredSent(sentok,target,tgtindex).strip()
	truesents.append(fixedsent)
	truemap[fixedsent] = []
	
	f.readline()
	f2.readline()
	cand = f.readline().strip()
	f2.readline()
	while len(cand)>0:
		truemap[fixedsent].append(cand)
		truecands.append(cand)
		cand = f.readline().strip()
		f2.readline()
	sent = f.readline().strip()
	sentok = f2.readline().strip()
f.close()
f2.close()

#Get language classes:
langhm = {}
f = open('C:/Users/ghpae/Google Drive/User Studies/Adaptive_Ranking/Scripts/language_hierarchy.txt')
for line in f:
	data = line.strip().split('\t')
	langhm[data[0].strip()] = data[1:]
f.close()

f = open('../../../corpora/datasets/surveymonkey_dataset_150118.txt')
line1 = f.readline().strip().split('\t')
line2 = f.readline().strip().split('\t')

ps = []
ptocount = {}
index = 15
token = line1[index]
p = token
ps.append(p)
while token!='NULL':
	ptocount[p] = 1
	index += 1
	token = line1[index]
	while len(token)==0:
		ptocount[p] += 1
		index += 1
		token = line1[index]
	p = token
	if p!='NULL':
		ps.append(p)

for i in range(6, len(line2)):
	c1 = line2[i].strip()
	c2 = truecands[i-6].strip()
	if c1!=c2:
		print 'nope'

		
#Build output file:
o1 = open('common20LS_victor.txt', 'w')
currid = -1
for line in f:
	data = line.strip().split('\t')
	if 109==len(data):
		currid += 1
		info = data[10:15]
		age = info[0]
		edu = info[1]
		prof = info[2].strip().split(' ')[0].strip()
		if len(info[4])>0:
			lang = info[4][0].upper()+info[4][1:]
		else:
			lang = info[3]
			
		answers = data[15:]
		candi = -1
		for sent, target, index in zip(truesents,truetargets,trueindexes):
			candmap = {}
			for cand in truemap[sent]:
				candi += 1
				index = int(answers[candi])
				candmap[cand] = index
			cands = sorted(candmap.keys(), key=candmap.__getitem__)
			newline_short = sent+'\t'+target.lower()+'\t'+str(index)
			for cand in cands:
				newline_short += '\t'+str(candmap[cand])+':'+cand
			newline_short += '\n'
			o1.write(newline_short.lower())
f.close()
o1.close()