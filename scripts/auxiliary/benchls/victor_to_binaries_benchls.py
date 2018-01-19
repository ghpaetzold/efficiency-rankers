

f = open('../corpora/datasets/benchls.txt')
o = open('../corpora/datasets/benchls_binary.txt', 'w')
for line in f:
	data = line.strip().split('\t')
	cands = data[3:]
	prefix = '\t'.join(data[:3])
	for i in range(0, len(cands)-1):
		for j in range(i+1, len(cands)):
			cand1 = cands[i]
			cand2 = cands[j]
			o.write(prefix+'\t'+cand1+'\t'+cand2+'\n')
f.close()
o.close()
