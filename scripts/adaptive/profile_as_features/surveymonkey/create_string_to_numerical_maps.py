import codecs

f = codecs.open('../../../../corpora/datasets/language_hierarchy.txt', encoding='utf8')
labels = ['age', 'lang', 'langg', 'langgg', 'edu', 'prof']
values = {}
for l in labels:
	values[l] = set([])

values['age'] = set(['10', '20', '30', '40', '50', '60', '70'])
values['edu'] = set(['Postgraduate', 'Secondary/High School', 'Undergraduate', 'Primary/Elementary School'])
values['prof'] = set(['C2', 'C1', 'B1', 'B2', 'A1', 'A2'])

for line in f:
	data = line.strip().split('\t')
	values['lang'].add(data[0].strip())
	values['langg'].add(data[1].strip())
	values['langgg'].add(data[2].strip())
f.close()

for label in labels:
	o = codecs.open('../../../../corpora/datasets/maps/'+label+'.txt', 'w', encoding='utf8')
	labels = sorted(list(values[label]))
	print labels
	for i, l in enumerate(labels):
		o.write(l+'\t'+str(i)+'\n')
	o.close()
