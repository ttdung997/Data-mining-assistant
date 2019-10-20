from apyori import apriori
import pandas as pd

transactions = [["Ga gion cay", "Khoai tay ran", "Com"], ["Ga gion cay", "Ga vien", "Coca"], ["Ga BBQ", "Ga vien", "Coca"],
["Ga BBQ", "Ga nugget", "Com"], ["Ga gion cay", "Ga nugget", "Coca"], ["Ga gion cay", "Khoai tay ran", "Milo"], ["Ga BBQ", "Ga nugget", "nuoc suoi"], ["Ga BBQ", "Khoai tay ran", "nuoc suoi"],
["Ga gion khong cay", "Khoai tay ran", "Milo"], ["Ga gion khong cay", "Khoai tay ran", "nuoc suoi"], ["Ga gion khong cay", "Khoai tay ran", "nuoc suoi"], ["Ga gion cay", "Khoai tay ran", "nuoc suoi"], ["Ga BBQ", "Ga nugget", "Coca"],
["Ga gion cay", "Khoai tay ran", "Coca"], ["Ga BBQ", "Ga vien", "Milo"], ["Ga gion cay", "Ga nugget", "Milo"], ["Ga gion khong cay", "Pho mai que", "Milo"], ["Ga BBQ", "Ga nugget", "Milo"],
["Ga gion khong cay", "Khoai tay ran", "Coca"], ["Ga gion cay", "Ga gion khong cay", "Ga nugget", "Coca"], ["Ga BBQ", "Ga gion khong cay", "Khoai tay ran", "Com"], ["Ga gion cay", "Ga BBQ", "Khoai tay ran", "nuoc suoi"],
["Ga gion cay", "Ga gion khong cay", "Ga nugget", "Coca"], ["Ga gion cay", "Ga gion khong cay", "Ga vien", "Com"], ["Ga gion khong cay", "Ga gion cay", "Ga vien", "Milo"], ["Ga gion khong cay", "Ga BBQ", "Ga nugget", "Milo"], 
["Ga BBQ", "Ga gion khong cay", "Khoai tay ran", "nuoc suoi"], ["Ga gion khong cay", "Ga gion cay", "Pho mai que", "Com"], ["Ga BBQ", "Ga gion khong cay", "Ga vien", "Coca"], ["Ga gion cay", "Ga gion khong cay", "Ga nugget", "Com"],
["Ga gion khong cay", "Ga gion cay", "Khoai tay ran", "Coca"], ["Ga gion khong cay", "Ga BBQ", "Khoai tay ran", "Milo"], ["Ga gion khong cay", "Ga BBQ", "Khoai tay ran", "Coca"], ["Ga gion khong cay", "Ga gion cay", "Ga vien", "Com"], 
["Ga gion cay", "Ga gion khong cay", "Ga nugget", "Com"], ["Ga gion khong cay", "Ga gion cay", "Pho mai que", "Coca"], ["Ga gion khong cay", "Ga gion cay", "Khoai tay ran", "Com"], ["Ga gion khong cay", "Ga BBQ", "Pho mai que", "Milo"],
["Ga BBQ", "Ga vien", "Coca", "nuoc suoi"], ["Ga gion cay", "Ga vien", "Tra chanh", "Milo"], ["Ga gion cay", "Ga vien", "Ga gion cay", "Com"], ["Ga gion cay", "Ga vien", "Coca", "Com"],
["Ga gion cay", "Pho mai que", "Milo", "nuoc suoi"], ["Ga BBQ", "Khoai tay ran", "Coca", "Com"], ["Ga BBQ", "Ga vien", "Coca", "Milo"], ["Ga gion khong cay", "Pho mai que", "Milo", "Coca"], 
["Ga gion cay", "Ga vien", "Milo", "Com"], ["Ga gion cay", "Pho mai que", "Tra chanh", "nuoc suoi"], ["Ga BBQ", "Ga vien", "Tra chanh", "Milo"], ["Ga gion cay", "Ga nugget", "Coca", "nuoc suoi"], ["Ga BBQ", "Khoai tay ran", "Milo", "Com"],
["Ga BBQ", "Pho mai que", "Tra chanh", "Coca"], ["Ga BBQ", "Ga vien", "Tra chanh", "Milo"], ["Ga gion cay", "Ga vien", "Coca", "Milo"], ["Ga BBQ", "Ga nugget", "Coca", "Milo"], ["Ga BBQ", "Ga vien", "Ga gion cay", "Com"],
["Ga gion cay", "Ga nugget", "Tra chanh", "Milo"], ["Ga nugget", "Pho mai que", "Tra chanh", "nuoc suoi"], ["Ga nugget", "Ga vien", "Milo", "nuoc suoi"], ["Pho mai que", "Khoai tay ran", "Tra chanh", "Milo"], ["Ga vien", "Ga nugget", "Tra chanh", "Coca"],
["Pho mai que", "Khoai tay ran", "Milo", "Coca"], ["Khoai tay ran", "Ga nugget", "Tra chanh", "nuoc suoi"], ["Khoai tay ran", "Ga vien", "Milo", "nuoc suoi"], ["Ga nugget", "Ga vien", "Milo", "Coca"], ["Ga vien", "Ga nugget", "Ga gion cay", "Com"],
["Pho mai que", "Ga vien", "Tra chanh", "Coca"], ["Ga nugget", "Pho mai que", "Milo", "Com"], ["Khoai tay ran", "Ga nugget", "Coca", "Com"], ["Khoai tay ran", "Pho mai que", "Milo", "Coca"], ["Ga vien", "Khoai tay ran", "Coca", "nuoc suoi"],
["Ga vien", "Khoai tay ran", "Milo", "Com"], ["Ga nugget", "Ga vien", "Milo", "Coca"], ["Pho mai que", "Ga vien", "Ga gion cay", "Com"], ["Khoai tay ran", "Ga vien", "Milo", "Coca"], ["Ga nugget", "Pho mai que", "Ga gion cay", "Milo"], 
["Ga gion cay", "Ga nugget", "Ga vien", "Coca", "Com"], ["Ga gion cay", "Ga vien", "Ga nugget", "Milo", "nuoc suoi"], ["Ga gion cay", "Ga vien", "Khoai tay ran", "Milo", "Coca"], ["Ga gion cay", "Khoai tay ran", "Ga nugget", "Milo", "Coca"],
["Ga gion khong cay", "Ga vien", "Ga nugget", "Milo", "Com"], ["Ga gion cay", "Pho mai que", "Khoai tay ran", "Ga gion cay", "Milo"], ["Ga gion khong cay", "Khoai tay ran", "Ga nugget", "Milo", "Coca"], ["Ga gion khong cay", "Ga nugget", "Khoai tay ran", "Coca", "nuoc suoi"],
["Ga gion cay", "Pho mai que", "Ga vien", "Milo", "nuoc suoi"], ["Ga gion khong cay", "Pho mai que", "Khoai tay ran", "Milo", "nuoc suoi"], ["Ga BBQ", "Ga vien", "Khoai tay ran", "Coca", "Com"], ["Ga BBQ", "Ga nugget", "Ga vien", "Tra chanh", "Coca"],
["Ga BBQ", "Ga nugget", "Khoai tay ran", "Coca", "Com"], ["Ga gion khong cay", "Ga nugget", "Ga vien", "Ga gion cay", "Coca"], ["Ga gion khong cay", "Ga vien", "Pho mai que", "Tra chanh", "Milo"], ["Ga gion khong cay", "Ga nugget", "Ga vien", "Tra chanh", "nuoc suoi"],
["Ga gion cay", "Ga vien", "Ga nugget", "Coca", "Milo"], ["Ga gion cay", "Ga vien", "Ga nugget", "Coca", "nuoc suoi"], ["Ga gion cay", "Ga nugget", "Khoai tay ran", "Milo", "Com"], ["Ga vien", "Ga nugget", "Khoai tay ran", "Milo"],
["Ga vien", "Khoai tay ran", "Ga nugget", "Com"], ["Khoai tay ran", "Pho mai que", "Ga vien", "Coca"], ["Pho mai que", "Ga vien", "Ga nugget", "Coca"], ["Khoai tay ran", "Ga nugget", "Ga vien", "Milo"], ["Ga nugget", "Ga vien", "Khoai tay ran", "Com"],
["Ga vien", "Pho mai que", "Khoai tay ran", "Com"], ["Pho mai que", "Ga vien", "Ga nugget", "nuoc suoi"], ["Ga nugget", "Pho mai que", "Ga vien", "Coca"], ["Ga vien", "Khoai tay ran", "Pho mai que", "nuoc suoi"], ["Pho mai que", "Ga vien", "Khoai tay ran", "Com"],
["Ga vien", "Khoai tay ran", "Pho mai que", "nuoc suoi"], ["Ga nugget", "Ga vien", "Khoai tay ran", "Com"], ["Ga vien", "Khoai tay ran", "Pho mai que", "Milo"], ["Ga vien", "Khoai tay ran", "Ga nugget", "Com"], ["Ga nugget", "Khoai tay ran", "Ga vien", "nuoc suoi"], 
["Khoai tay ran", "Pho mai que", "Ga vien", "Com"], ["Khoai tay ran", "Ga vien", "Pho mai que", "nuoc suoi"], ["Ga vien", "Pho mai que", "Khoai tay ran", "nuoc suoi"], ["Ga nugget", "Ga vien", "Khoai tay ran", "Tra chanh", "Milo"], ["Khoai tay ran", "Pho mai que", "Ga vien", "Ga gion cay", "Com"],
["Khoai tay ran", "Ga nugget", "Ga vien", "Milo", "Com"], ["Khoai tay ran", "Ga vien", "Ga nugget", "Ga gion cay", "Coca"], ["Khoai tay ran", "Pho mai que", "Ga nugget", "Tra chanh", "Milo"], ["Pho mai que", "Ga vien", "Ga nugget", "Ga gion cay", "Com"], 
["Khoai tay ran", "Pho mai que", "Ga vien", "Ga gion cay", "Coca"], ["Ga nugget", "Ga vien", "Pho mai que", "Milo", "Com"], ["Pho mai que", "Ga nugget", "Khoai tay ran", "Tra chanh", "nuoc suoi"], ["Khoai tay ran", "Pho mai que", "Ga vien", "Coca", "Milo"], 
["Khoai tay ran", "Ga vien", "Ga nugget", "Coca", "Com"], ["Ga nugget", "Ga vien", "Pho mai que", "Tra chanh", "Coca"], ["Ga vien", "Khoai tay ran", "Pho mai que", "Coca", "Milo"], ["Pho mai que", "Ga nugget", "Ga vien", "Milo", "Com"],
["Ga vien", "Ga nugget", "Pho mai que", "Coca", "nuoc suoi"], ["Pho mai que", "Ga vien", "Ga nugget", "Tra chanh", "nuoc suoi"], ["Ga vien", "Khoai tay ran", "Ga nugget", "Milo", "Com"], ["Pho mai que", "Ga vien", "Khoai tay ran", "Tra chanh", "Coca"],
["Khoai tay ran", "Ga vien", "Ga nugget", "Ga gion cay", "Com"]]

df = pd.DataFrame(transactions)
df.to_csv("20150722.csv",index=False, header=False)
#tuy chinh cac tham so
# min_length = 1, min_support = 0.2, min_confidence = 0.2, min_lift = 3
results = list(apriori(transactions,min_support = 0.1, min_confidence = 0.5))

# lay cac ket qua tu danh sach tra ve
listRules = [[results[i][2][0][0],results[i][2][0][1],results[i][1],results[i][2][0][2],results[i][2][0][3]] for i in range(0,len(results)) if len(results[i][0]) > 1]

for rule in listRules:
	print(str(list(rule[0]))+"->"+str(list(rule[1])))
	print("sup: "+str(rule[2])+",conf: "+str(rule[3])+ ",lift: "+str(rule[4]))