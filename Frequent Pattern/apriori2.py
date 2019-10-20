
# su dung thu vien de goi thuat toan apriori
from efficient_apriori import apriori

import pandas as pd
# du lieu giao dich mau
transactions = []
data = [line.rstrip('\n') for line in open('store_data.csv')]

for line in data:
	line= line.split(",")
	transactions.append(tuple(line))

# print(transactions)
#goi thuat toan de xay dung luat, dieu chinh cac tham so min_support, min_confidence
itemsets, rules = apriori(transactions, min_support=0.05,  min_confidence=0.1)

# print(rules) 

# Them rang buoc cho luat
# lhs: luat trai, rhs: luat phai

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules, key=lambda rule: rule.lift):
	print(rule) # In luat va cac cac chi so danh gia