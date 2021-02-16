import numpy as np
def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
   
    points = {i : 0 for i in range(n)}
    centers = []
    centers.append(generator.randint(0, n))
    points.pop(centers[-1])

    while len(centers) < n_cluster:
        for p in points.keys():
            points[p] = np.amin(np.array([np.linalg.norm(x[p] - x[c]) for c in centers]))
        idx_list = np.array(list(points.keys()))
        dist_list = np.array(list(points.values()))
        dist_list = dist_list / np.sum(dist_list)
        r = generator.rand()
        d = 0
        idx = 0
        for i in range(len(dist_list)):
            d += dist_list[i]
            if (d >= r):
                idx = i
                break
        
        centers.append(idx_list[idx])
        points.pop(centers[-1])

    return centers



n_cluster = 3
x = np.array([(-10,1), (-10,1), (-10,1), (-10,1), (-10,1), (10,1), (10,1), (10,1), (10,1), (10,1), (0, -10)])
n = len(x)
centers = get_k_means_plus_plus_center_indices(n, n_cluster, x)
print(centers)

