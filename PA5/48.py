import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################


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
    dist = []
    while len(centers) < n_cluster:
        d2_last_center = np.array([np.square(np.linalg.norm(p - x[centers[-1]])) for p in x])
        dist = d2_last_center if len(dist) == 0 else np.minimum(d2_last_center, dist)
        
        d_percent = dist/sum(dist)
        r = generator.rand()
        
        percent = 0
        idx = 0
        for i in range(len(d_percent)):
            percent += d_percent[i]
            if (percent >= r):
                idx = i
                break
        centers.append(idx)
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
        
        centroids = np.array([x[self.centers[j]] for j in range(self.n_cluster)])
        y = np.zeros(N)
        distort = np.sum([np.sum((x[y == i] - centroids[i])) for i in range(self.n_cluster)]) / N

        n_iter = 0
        
        while n_iter < self.max_iter:
            y = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1))**2), axis=2), axis=0)
            distort_n = np.sum([np.sum((x[y == i] - centroids[i])) for i in range(self.n_cluster)]) / N
            if abs(distort - distort_n) <= self.e:
                break

            distort = distort_n
            centroids_n = np.array([np.mean(x[y == cluster_ind], axis=0) for cluster_ind in range(self.n_cluster)])
            centroids_n[np.where(np.isnan(centroids_n))] = centroids[np.where(np.isnan(centroids_n))]
            centroids = centroids_n
            n_iter = n_iter + 1
            
        return centroids, y, self.max_iter
        


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
        centroids, membership, n_iter = kmeans.fit(x)

        votes = [{} for k in range(self.n_cluster)]
  
        for l, r in zip(y, membership):
            votes[r][l] = votes[r][l] + 1 if l in votes[r].keys() else 1

        labels = []
        for v in votes:
            l = max(v, key = v.get) if v else 0
            labels.append(l)

        centroid_labels = np.array(labels)
        
        
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
