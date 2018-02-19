import os

#scripts = ['boundary.py']
scripts = ['hybridwithfeat.py']
combinations = [(2, 50), (4, 100), (8, 200)]

for script in scripts:
		for prop in range(10, 100, 20):
			for comb in combinations:
				comm = 'nohup python -u '+script+' '+str(prop)+' 5 5000 '+str(comb[0])+' '+str(comb[1])+' &'
				os.system(comm)
