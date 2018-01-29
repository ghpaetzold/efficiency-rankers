import os

scripts = ['boundary.py']
#scripts = ['ridge.py']
modes = ['normal', 'individual', 'group-age', 'group-lang', 'group-langg', 'group-langgg', 'group-edu', 'group-prof']

for script in scripts:
	for mode in modes:
		comm = 'nohup python -u '+script+' '+mode+' &'
		os.system(comm)
