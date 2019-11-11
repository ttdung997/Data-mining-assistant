import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans

# Tạo bộ dữ liệu thử nghiệm
X, y = make_blobs(
   n_samples=150, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)

# Dữ liệu
print(X)

# gọi thư viện K-mean
# n_clusters: Số cụm chia dữ liệu
# n_init: Số lần tạo random cụm
# max_iter: Số lần lặp mỗi cụm


km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

# quá trình học
y_km = km.fit_predict(X)



#  biểu diện quá trình phân cụm


plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# vẽ các tâm cho mỗi cụm
# km.cluster_centers_: danh sách tâm cụm
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

# Kiểm tra một điểm mới bất kỳ
print(km.predict([[1,0]]))
