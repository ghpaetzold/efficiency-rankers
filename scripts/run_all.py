import os

scripts = ['ridge.py', 'boundary.py']
#scripts = ['boundary.py']
datasets = ['benchls.txt']

for script in scripts:
	for dataset in datasets:
		for prop in range(20, 100, 20):
		#for prop in [20,80]:
			comm = 'nohup python -u '+script+' ../corpora/datasets/'+dataset+' '+str(prop)+' 5 5 &'
			os.system(comm)
