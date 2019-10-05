
# su dung thu vien de goi thuat toan apriori
from efficient_apriori import apriori

import pandas as pd
# du lieu giao dich mau
transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]


#goi thuat toan de xay dung luat, dieu chinh cac tham so min_support, min_confidence

itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)

# print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]

# Them rang buoc cho luat
# lhs: luat trai, rhs: luat phai

# rules_rhs = filter(lambda rule: len(rule.lift) == 2 and len(rule.rhs) == 1, rules)


rules_rhs = filter(lambda rule: rule.lift >1.2, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
	print(rule) # In luat va cac cac chi so danh gia
