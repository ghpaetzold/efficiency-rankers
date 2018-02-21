import os

scripts = ['boundary.py']
#scripts = ['ridge.py']
#modes = ['normal', 'individual', 'group-age', 'group-lang']
modes = ['group-langg', 'group-langgg', 'group-edu', 'group-prof']
#modes = ['normal', 'individual', 'group-age', 'group-lang', 'group-langg', 'group-langgg', 'group-edu', 'group-prof']

#modes = ['normal']

for script in scripts:
	for mode in modes:
		for candsize in [5,10,20,30,40]:
			comm = 'nohup python -u '+script+' '+str(candsize)+' 2 500 '+mode+' &'
			os.system(comm)
