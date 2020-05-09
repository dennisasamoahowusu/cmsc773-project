import cotools as co
from pprint import pprint
import os
import sys

downloaded = True # change me if you havent downloaded the data

if not downloaded:
    co.download(dir = 'data', match = "2020-04-10", regex = True)
    
pprint(os.listdir('data'))

data = co.Paperset('data/custom_license')
print(str(sys.getsizeof(data))+' bytes')

print(f"{len(data)} papers")

print()
print("How data[index] looks like:")
pprint(data[13])

print()
print("How text looks like")
pprint(co.text(data[13]))

print()
print("How abstract looks like")
try:
	pprint(co.abstract(data[13]))
except KeyError:
	print("Abstract Not Found") 

#pprint(co.abstracts(data[14:18]))

#abstracts = data.abstracts()
#pprint(abstracts)

## finding abstracts
print()
print("Finding abstracts")
#for x in data[100:5000]:
#	try:
#		pprint(co.abstract(x))
#	except KeyError:
#		print("Abstract Not Found")
