import os

scripts = ['ridge.py', 'boundary.py', 'mlp.py']
#scripts = ['ridge.py']
datasets = ['benchls.txt']
datasets = ['nnseval.txt']

for script in scripts:
	for dataset in datasets:
		for prop in range(10, 100, 10):
#		for prop in [f for f in range(10, 100, 10) if f not in set(range(20, 100, 20))]:
		#for prop in range(20, 100, 20):
		#for prop in [90]:
			comm = 'nohup python -u '+script+' ../../corpora/datasets/'+dataset+' '+str(prop)+' 5 5 &'
			os.system(comm)
