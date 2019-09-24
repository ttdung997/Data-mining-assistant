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

length = len(transactions)
# In ra cac cap cofident trong mau du lieu
# find_frequent_patterns(du lieu, minsup)
patterns = pyfpgrowth.find_frequent_patterns(transactions, length/2)

# In ra du lieu luat ket hop
# find_frequent_patterns(patterns, minconf)
rules = pyfpgrowth.generate_association_rules(patterns, 0.5)

#bieu dien luat ket hop
keys = rules.keys()
for key in keys:
	print(str(key)+" -> "+str(rules[key][0])+ "("+str(float(patterns[key])/length)+", "+str(rules[key][1])+")")

