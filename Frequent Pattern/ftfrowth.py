import pyfpgrowth

# Du lieu giao dich voi cac mat hang danh so tu 1->5
transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]

# dem do ho cua item trong cum 
def support_count(rhs):
	count =0
	rhs = set(rhs)
	for j in transactions:
		j = set(j)
		if(rhs.issubset(j)):
			count = count + 1
	return count



length = len(transactions)
print(length)
# In ra cac cap cofident trong mau du lieu
# find_frequent_patterns(du lieu, minsup)
patterns = pyfpgrowth.find_frequent_patterns(transactions, length/2)
print(patterns)
# In ra du lieu luat ket hop
# find_frequent_patterns(patterns, minconf)
rules = pyfpgrowth.generate_association_rules(patterns, 0.5)

#bieu dien luat ket hop

keys = rules.keys()
for key in keys:
	#tinh sub cua 2 thanh phan trai, phai trong luat
	sup_right =support_count(rules[key][0])/float(length)
	sup_left = support_count(key)/float(length)

	# bieu dien luat
	print(str(key)+" -> "+str(rules[key][0]))
	print("sup: "+str(float(patterns[key+rules[key][0]])/length)+",conf: "+str(rules[key][1]))
	#cach tinh 1
	print("lift: "+str(rules[key][1]/sup))
	# canh tinh 2
	print("lift: "+str(float(patterns[key+rules[key][0]])/length/(sup*sup2)))

