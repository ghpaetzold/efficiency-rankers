

f = open('../corpora/datasets/benchls.txt')
for line in f:
	data = line.strip().split('\t')
	print data
f.close()