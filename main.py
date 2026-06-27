from k_means import k_means

points = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
labels, centroids, iterations = k_means(points, k=2, seed=42)

print(labels)       # метка кластера для каждой точки
print(centroids)    # финальные центроиды
print(iterations)   # число итераций