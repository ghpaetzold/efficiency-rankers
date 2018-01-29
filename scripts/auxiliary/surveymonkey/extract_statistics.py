import numpy as np
import matplotlib.pyplot as plt

ages = []
for column in range(0, 7):
	f = open('../../../corpora/datasets/SurveyMonkey_150118_Data_With_IDS.txt')
	values = []
	counts = {}
	for line in f:
		data = line.strip().split('\t')
		if column==1:
			ages.append(int(data[column]))
			data[column] = int(10*(float(data[column])//10))
		values.append(data[column])
		if data[column] not in counts:
			counts[data[column]] = 0
		counts[data[column]] += 1

	labelv = sorted(counts.keys(), key=counts.__getitem__, reverse=True)
	countv = [counts[c] for c in labelv]
	
	print data[column], len(counts.keys())

	plt.clf()
	#plt.pie(countv, labels=labelv, autopct='%1.0f%%')
	plt.pie(countv, autopct='%1.0f%%', pctdistance=1.1, shadow=True, startangle=45)
	lgd = plt.legend(labelv, loc="right", bbox_to_anchor=(1.1, 0.5))
	#plt.tight_layout()
	plt.axis('equal')
	#plt.savefig(str(column)+'.png', dpi=300, additional_artists=[lgd], bbox_inches="tight")
	plt.show()

	f.close()
	
print max(ages), min(ages)