from __future__ import print_function

# map 1: x  ->  [0, 100]  ->  [30, 100]
# map 2: x  ->  [0, 50]  ->  [15, 50]
# map 11:x  ->  [0, 30]  ->  [10, 30]

def func1(x):
    if x > 100:
        x = 100
    return int(x * 0.7) + 30

def func2(x):
    if x > 50:
        x = 50
    return int(x * 0.7) + 15

def func3(x):
    if x > 30.0:
        x = 30.0
    return x * 0.7 + 10.0

f = open("example.txt", "r")
line = f.readline()
while line:
    line = line.rstrip()
    fields = line.split(' ')
    label = fields[0]
    feats = fields[1:]
    print ("%d" % int(label), end='')
    for feat in feats:
        [i, v] = feat.split(':')
        index = int(i)
        if index % 12 == 1:
            val = func1(int(v))
            print(" %d:%d" % (index, val), end='')
        elif index % 12 == 2:
            val = func2(int(v))
            print(" %d:%d" % (index, val), end='')
        elif index % 12 == 11:
            val = func3(float(v))
            print(" %d:%f" % (index, val), end='')
    print("\n", end='')
    line = f.readline()
f.close()
