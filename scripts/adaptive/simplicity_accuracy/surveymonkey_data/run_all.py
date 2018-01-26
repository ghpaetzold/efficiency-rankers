import os

scripts = ['boundary.py']
#scripts = ['ridge.py']
#modes = ['normal', 'individual', 'group-age', 'group-lang']
modes = ['group-langg', 'group-langgg', 'group-edu', 'group-prof']

#modes = ['normal']

for script in scripts:
	for mode in modes:
		for prop in range(10, 100, 10):
			comm = 'nohup python -u '+script+' '+str(prop)+' 5 500 '+mode+' &'
			os.system(comm)
