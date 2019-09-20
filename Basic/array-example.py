# khai bao list
x = []

#tham phan tu vao list
for i in range(0,5):
	x.append(i)

print(x)
# [0, 1, 2, 3, 4]

# khai bao array
X = [1,2,3,4,5]

# tao mot array moi tu array cu
Y =[x+1 for x in X]

print(X)
# [1, 2, 3, 4, 5]

print(Y)
# [2, 3, 4, 5, 6]
