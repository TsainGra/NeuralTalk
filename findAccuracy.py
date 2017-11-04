import csv

acc_val = []

with open('accuracy.txt', 'rU') as f:
	acc = csv.reader(f)
	for a in acc:
		acc_val.append(a)

print "Size of dataset:", len(acc_val)

values = []
for i in range(0,len(acc_val)):
	values.append(acc_val[i][0])

print "Highest value:", max(values)
print "Index of highest value:", values.index(max(values))

# Now choosing highest accuracy among the saved weights.
# Weights saved for every thousandth iteration.
big = 0.0
big_ind = 0
for i in range(0,296734):
	if i%1000 == 0:
		if values[i]>big:
			big = values[i]
			big_ind = i

print "Highest accuracy of saved weight:", big
print "Index:",big_ind
"""
Following code does not work
arr_ind = []
arr_val = []
for i in range(0,len(values)):
	if i%1000 == 0:
		arr_ind.append(i)
		arr_val.append(values[i][0])

print max(arr_val)
print arr_val.index(max(arr_val))
"""
