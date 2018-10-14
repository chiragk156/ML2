import random

# Data reading from file
# Change File path here
file = open('linregdata','r')
data = file.read().split('\n')
file.close()
if '' in data:
	data.remove('')

n=len(data)

for i in range(0,n):
	data[i]=data[i].split(',')

# Randomly partitioning the data
r=random.sample(range(0, n), n)
test_n = round(0.2*n)

file = open('testdata.csv','w')
for i in range(0,test_n):
	if data[r[i]][0]=='F':
		file.write('1,0,0,'+str(data[r[i]][1])+','+str(data[r[i]][2])+','+str(data[r[i]][3])+','+str(data[r[i]][4])+','+str(data[r[i]][5])+','+str(data[r[i]][6])+','+str(data[r[i]][7])+','+str(data[r[i]][8])+'\n')
	elif data[r[i]][0]=='I':
		file.write('0,1,0,'+str(data[r[i]][1])+','+str(data[r[i]][2])+','+str(data[r[i]][3])+','+str(data[r[i]][4])+','+str(data[r[i]][5])+','+str(data[r[i]][6])+','+str(data[r[i]][7])+','+str(data[r[i]][8])+'\n')
	elif data[r[i]][0]=='M':
		file.write('0,0,1,'+str(data[r[i]][1])+','+str(data[r[i]][2])+','+str(data[r[i]][3])+','+str(data[r[i]][4])+','+str(data[r[i]][5])+','+str(data[r[i]][6])+','+str(data[r[i]][7])+','+str(data[r[i]][8])+'\n')
file.close()

file = open('trainingdata.csv','w')
for i in range(test_n,n):
	if data[r[i]][0]=='F':
		file.write('1,0,0,'+str(data[r[i]][1])+','+str(data[r[i]][2])+','+str(data[r[i]][3])+','+str(data[r[i]][4])+','+str(data[r[i]][5])+','+str(data[r[i]][6])+','+str(data[r[i]][7])+','+str(data[r[i]][8])+'\n')
	elif data[r[i]][0]=='I':
		file.write('0,1,0,'+str(data[r[i]][1])+','+str(data[r[i]][2])+','+str(data[r[i]][3])+','+str(data[r[i]][4])+','+str(data[r[i]][5])+','+str(data[r[i]][6])+','+str(data[r[i]][7])+','+str(data[r[i]][8])+'\n')
	elif data[r[i]][0]=='M':
		file.write('0,0,1,'+str(data[r[i]][1])+','+str(data[r[i]][2])+','+str(data[r[i]][3])+','+str(data[r[i]][4])+','+str(data[r[i]][5])+','+str(data[r[i]][6])+','+str(data[r[i]][7])+','+str(data[r[i]][8])+'\n')
file.close()