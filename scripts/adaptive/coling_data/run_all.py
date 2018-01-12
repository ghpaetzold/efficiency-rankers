import os

scripts = ['boundary.py']
scripts = ['ridge.py']
modes = ['normal', 'individual', 'group-age', 'group-lang', 'group-edu', 'group-prof']

for script in scripts:
	for mode in modes:
		for prop in range(10, 100, 10):
			comm = 'nohup python -u '+script+' '+str(prop)+' 5 500 '+mode+' &'
			os.system(comm)
