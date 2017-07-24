import numpy as np


d = {}

data = np.load('training_data.npy')

for i in data:
    t = ''.join( map(str, i[1] ))
    d[t] = 1 if t not in d else d[t] + 1

a = []
b = []
c = []
d = []


for i in data:
    if i[1][0] == 1 and len(a) < 34:
        a.append(i)
    elif i[1][1] == 1 and len(b) < 34:
        b.append(i)
    elif i[1][2] == 1 and len(c) < 34:
        c.append(i)
    elif i[1][3] == 1 and len(d) < 34:
        d.append(i)


e = a + b + c + d

print len(e)

d = {}

for i in e:
    t = ''.join( map(str, i[1] ))
    d[t] = 1 if t not in d else d[t] + 1

print d

np.random.shuffle(e)

np.save('balanaced_training_data.npy',e)