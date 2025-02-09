import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def dist_from_centroids(point, x, d2):
    res1 = []
    for i in range(len(x)):
        if len(d2)==0:
            res=[]
        else:    
            res = [d2[i]]
        res.append(np.square(np.linalg.norm(x[i] - x[point])))
        res1.append(min(res))    
    return res1

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # TODO: implement the Kmeans++ initialization
    ###############################################
    centers = []
    centers.append(generator.randint(0, n))
    d = []
    while len(centers) < n_cluster:
        d2 = dist_from_centroids(centers[-1], x, d)
        dp = d2/sum(d2)
        r = generator.rand()
        dist = 0
        idx = 0
        for i in range(len(dp)):
            dist += dp[i]
            if (dist >= r):
                idx = i
                break
        
        centers.append(idx)
        d = d2

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        def compute_distortion(centroids, x, y):
            distort = np.sum([np.sum((x[y == i] - centroids[i])) for i in range(self.n_cluster)])
            return distort/x.shape[0]

        centroids = np.zeros((self.n_cluster, D))
        for j in range(len(self.centers)):
            centroids[j] = x[self.centers[j]]
        y = np.zeros(N)
        distort = compute_distortion(centroids, x, y)
        n_iter = 0
        
        while n_iter < self.max_iter:
            y = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1))**2), axis=2), axis=0)
            distort_n = compute_distortion(centroids, x, y)
            if abs(distort - distort_n) <= self.e:
                break

            distort = distort_n
            z, a = 0, 0
            centroids_n = np.array([np.mean(x[y == cluster_ind], axis=0) for cluster_ind in range(self.n_cluster)])
            a = z
            centroids_n[np.where(np.isnan(centroids_n))] = centroids[np.where(np.isnan(centroids_n))]
            centroids = centroids_n
            z += 2
            n_iter += 1
            
        return (centroids, y, self.max_iter)
        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, _ = kmeans.fit(x)

        votes = [{} for k in range(self.n_cluster)]
  
        for l, r in zip(y, membership):
            votes[r][l] = votes[r][l] + 1 if l in votes[r].keys() else 1

        centroid_labels = np.array()
        for v in votes:
            l = max(v, key = v.get) if v else 0
            centroid_labels.append(l)
            # if not v:
            #     centroid_labels.append(0)
            # else:
            #     centroid_labels.append(max(v, key=v.get))

        # centroid_labels = np.array(centroid_labels)
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        l2_norm = np.sum(((x - np.expand_dims(self.centroids,axis=1))**2), axis=2)
        r = np.argmin(l2_norm, axis=0)
        return self.centroid_labels[r]




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    N, M, C = image.shape
    data = image.reshape(N * M, C)
    r = np.argmin(np.sum(((data - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2), axis=0)
    new_im = code_vectors[r].reshape(N, M, C)
    
    return new_im
