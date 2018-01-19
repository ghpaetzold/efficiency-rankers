def getIndex(target, sent):
	newtgt = target.lower().replace(' ', '_')
	newsent = sent.lower().replace(target.lower(), newtgt)
	tokens = newsent.split(' ')
	for i, token in enumerate(tokens):
		if newtgt==token:
			return i
	return -1
	
def getUnderscoredSent(sent, target, tgtindex):
	newsent = sent.lower().replace(target.lower(), target.lower().replace(' ', '_'))
	tokens = newsent.split(' ')
	return ' '.join(tokens[:tgtindex]).strip()+' _____________ '+' '.join(tokens[tgtindex+1:]).strip()

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
		for sent, target, tgtindex in zip(truesents,truetargets,trueindexes):
			candmap = {}
			for cand in truemap[sent]:
				candi += 1
				index = int(answers[candi])
				candmap[cand] = index
			cands = sorted(candmap.keys(), key=candmap.__getitem__)
			for i in range(0, len(cands)-1):
				for j in range(i+1, len(cands)):
					newline = str(currid)+'\t'+str(age)+'\t'+lang+'\t'+langhm[lang][0]+'\t'+langhm[lang][1]+'\t'+edu+'\t'+prof+'\t'+sent+'\t'+target.lower()+'\t'+str(tgtindex)
					newline += '\t'+cands[i]+'\t'+cands[j]+'\t'+cands[i]
					newline += '\n'
					o.write(newline)
f.close()
o.close()

f = open('SurveyMonkey_New_Data_With_IDS.txt')
for line in f:
	data = line.strip().split('\t')
	sent = data[7].split(' ')
	index = int(data[9])
	if sent[index]!='_____________':
		print data
f.close()
