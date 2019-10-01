from apyori import apriori

transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]
#tuy chinh cac tham so
# min_length = 1, min_support = 0.2, min_confidence = 0.2, min_lift = 3
results = list(apriori(transactions,min_support = 0.2, min_confidence = 0.2))

# lay cac ket qua tu danh sach tra ve
listRules = [[results[i][2][0][0],results[i][2][0][1],results[i][1],results[i][2][0][2],results[i][2][0][3]] for i in range(0,len(results)) if len(results[i][0]) > 1]

for rule in listRules:
	print(str(list(rule[0]))+"->"+str(list(rule[1])))
	print("sup: "+str(rule[2])+",conf: "+str(rule[3])+ ",lift: "+str(rule[4]))