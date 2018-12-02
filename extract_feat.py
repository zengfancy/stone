
f = open("example.txt", "r")
line = f.readline()
while line:
    line = line.rstrip()
    fields = line.split(' ')
    label = fields[0]
    feats = fields[1:]
#    print "label:", label
    for feat in feats:
        [index, val] = feat.split(':')
        index = int(index)
        val = float(val)
        if (index - 1) % 12 == 0:
            print "%d:%f" % (index, val)
#        print "index: %s, val: %s" % (index, val)
    line = f.readline()
f.close()
