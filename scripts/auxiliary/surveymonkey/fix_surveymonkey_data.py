def getIndex(target, sent):
	tokens = sent.lower().split(' ')
	for i, token in enumerate(tokens):
		if target.lower()==token:
			return i
	return -1
	
def getUnderscoredSent(sent, tgtindex):
	tokens = sent.split(' ')
	return ' '.join(tokens[:tgtindex]).strip()+' _____________ '+' '.join(tokens[tgtindex+1:]).strip()

#Get instances:
f = open('C:/Users/ghpae/Google Drive/User Studies/Adaptive_Ranking/Corpora/SCC instances/adjusted_selected_instances.txt')
truemap = {}
sent = f.readline().strip()
count = 0
truesents = []
truetargets = []
trueindexes = []
truecands = []
while len(sent)>0:
	count += 1
	target = f.readline().strip()
	
	truetargets.append(target)
	tgtindex = getIndex(target, sent)
	trueindexes.append(tgtindex)
	fixedsent = getUnderscoredSent(sent,tgtindex).strip()
	truesents.append(fixedsent)
	truemap[fixedsent] = []
	
	f.readline()
	cand = f.readline().strip()
	while len(cand)>0:
		truemap[fixedsent].append(cand)
		truecands.append(cand)
		cand = f.readline().strip()
	sent = f.readline().strip()
f.close()

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
o = open('SurveyMonkey_New_Data_With_IDS.txt', 'w')
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
			for i in range(0, len(cands)-1):
				for j in range(i+1, len(cands)):
					newline = str(currid)+'\t'+str(age)+'\t'+lang+'\t'+langhm[lang][0]+'\t'+langhm[lang][1]+'\t'+edu+'\t'+prof+'\t'+sent+'\t'+target.lower()+'\t'+str(index)
					newline += '\t'+cands[i]+'\t'+cands[j]+'\t'+cands[i]
					newline += '\n'
					o.write(newline)
f.close()
o.close()