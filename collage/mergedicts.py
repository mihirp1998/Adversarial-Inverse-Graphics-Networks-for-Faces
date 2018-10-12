import pickle
from itertools import chain
from collections import defaultdict

a = pickle.load(open('fakeBrealA.p','rb'))
a1 = pickle.load(open('fakeBrealA1.p','rb'))

finaldict = defaultdict()
for k,v in chain(a.items(),a1.items()):
	if k not in finaldict.keys():
		finaldict[k] = v
	else:
		print("same")			
for k,v in finaldict.items():
	if len(v) != 2:
		finaldict.pop(k)
		print(k,v)
pickle.dump(finaldict,open('human2avatar.p','wb'))


import pickle
from itertools import chain
from collections import defaultdict

a = pickle.load(open('fakeArealB.p','rb'))
a1 = pickle.load(open('fakeArealB1.p','rb'))

finaldict = defaultdict()
for k,v in chain(a.items(),a1.items()):
	if k not in finaldict.keys():
		finaldict[k] = v
	else:
		print("same")			
for k,v in finaldict.items():
	if len(v) != 2:
		finaldict.pop(k)
		print(k,v)
pickle.dump(finaldict,open('avatar2human.p','wb'))		
# print(finaldict)		