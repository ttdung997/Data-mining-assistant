
# su dung thu vien de goi thuat toan apriori
from efficient_apriori import apriori

# du lieu giao dich mau
transactions = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]

#goi thuat toan de xay dung luat, dieu chinh cac tham so min_support, min_confidence

itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)

print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]