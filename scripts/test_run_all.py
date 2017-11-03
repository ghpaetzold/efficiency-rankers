import os

scripts = ['test_ridge.py']
datasets = ['benchls_binary.txt']

for script in scripts:
	for dataset in datasets:
		for prop in range(20, 100, 20):
			comm = 'nohup python -u '+script+' ../corpora/datasets/'+dataset+' '+str(prop)+' 5 200 &'
			os.system(comm)
