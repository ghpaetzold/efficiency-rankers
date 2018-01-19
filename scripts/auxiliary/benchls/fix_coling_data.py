def getUnique(line):
	unique = '\t'.join(line[:4])
	return unique

path = '../../corpora/datasets/COLING_SR_Annotations.txt'
f = open(path)
o = open('../../corpora/datasets/COLING_SR_Annotations_with_IDs.txt', 'w')
line = f.readline().strip().split('\t')
unique = getUnique(line)
curr = unique
id = 0
while len(line)>1:
	unique = getUnique(line)	
	if unique!=curr:
		id += 1
		curr = unique
	if 'The words are equally simple or equally complex.' not in line:
		o.write(str(id)+'\t'+'\t'.join(line)+'\n')
	line = f.readline().strip().split('\t')
f.close()
o.close()
