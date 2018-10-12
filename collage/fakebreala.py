from os import listdir
from os.path import isfile, join
from itertools import chain
from collections import defaultdict
import pickle
mypath = "till620"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(len(onlyfiles))
fakeB = []
realA = []
realA_frame = []
fakeB_frame = []
for i in onlyfiles:
	if 'real_A' in i or 'rec_A' in i:
		if 'frame' in i :
			realA_frame.append(i)
		else:
			realA.append(i)
	elif 'fake_B' in i:
		if 'frame' in i :
			fakeB_frame.append(i)		
		else:
			fakeB.append(i)
	else:
		print(i)	
print(len(realA))
print(len(realA_frame))
print(len(fakeB))			
print(len(fakeB_frame))		
realA_dict = {}
for i in realA:
	if i[:6].isdigit():
		realA_dict[i[:6]] = i	
fakeB_dict = {}
for i in fakeB:
	if i[:6].isdigit():
		fakeB_dict[i[:6]] = i	
# print(realA_dict.keys())	
# print(realA_dict['102091'])
finaldict = defaultdict(list)
for k,v in chain(realA_dict.items(),fakeB_dict.items()):
	finaldict[k].append(v)
for k,v in finaldict.items():
	if len(v) != 2:
		finaldict.pop(k)
		# print(k,v)	
# print(finaldict)		
# pickle.dump(finaldict,open('finaldict.p','wb'))



realA_dict = {}
for i in realA_frame:
	if i[5:10].isdigit():
		realA_dict[i[5:10]] = i	
fakeB_dict = {}
for i in fakeB_frame:
	if i[5:10].isdigit():
		fakeB_dict[i[5:10]] = i	
# print(realA_dict.keys())	
# print(realA_dict['102091'])
# finaldict = defaultdict(list)

for k,v in chain(realA_dict.items(),fakeB_dict.items()):
	finaldict[k].append(v)
for k,v in finaldict.items():
	if len(v) != 2:
		finaldict.pop(k)
		# print(k,v)	
print(len(finaldict.keys()))		
pickle.dump(finaldict,open('fakeBrealA1.p','wb'))