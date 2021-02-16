import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    b = 1
    # tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    # fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    # fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    correctPos = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    predictPos = sum([y == 1 for y in predicted_labels])
    realPos = sum([x == 1 for x in real_labels])

    if (predictPos != 0 and realPos != 0):
        precision = correctPos / predictPos
        recall = correctPos / realPos
        if b * b * precision + recall != 0:
            f1 = (1 + b * b) * (precision * recall) / (b * b * precision + recall)
        else:
            f1 = 0
    else:
        f1 = 0
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p = 3
        d = [abs(x - y) ** p for x, y in zip(point1, point2)]
        a = sum(d)**(1/p)

        return a

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        d = [abs(x - y) ** 2 for x, y in zip(point1, point2)]
        d = np.sqrt(sum(d))

        return d

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p1norm = np.linalg.norm(point1, 2)
        p2norm = np.linalg.norm(point2, 2)
        if not p1norm or not p2norm:
            return 1
        
        d = 1 - np.dot(point1, point2) / (p1norm * p2norm)
        return d

        



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        best_f1 = 0
        for name, func in distance_funcs.items():
            for k in range(1, 30, 2):
                model = KNN(k, func)
                model.train(x_train, y_train)
                valid_f1 = f1_score(y_val, model.predict(x_val))
                if valid_f1 > best_f1:
                    self.best_distance_function = name
                    self.best_k = k
                    best_f1 = valid_f1
                    self.best_model = model


    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        best_f1 = 0
        for scaling_name, scaling_func in scaling_classes.items():
            scaler = scaling_func()
            x_train_scaled = scaler(x_train)
            x_val_scaled = scaler(x_val)
            for name, func in distance_funcs.items():
                for k in range(1, 30, 2):
                    model = KNN(k, func)
                    model.train(x_train_scaled, y_train)
                    valid_f1 = f1_score(y_val, model.predict(x_val_scaled))
                    if valid_f1 > best_f1:
                        self.best_distance_function = name
                        self.best_k = k
                        best_f1 = valid_f1
                        self.best_model = model
                        self.best_scaler = scaling_name



class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norm = []
        for data in features:
            if all(x == 0 for x in data):
                norm.append(data)
            else:
                scale = sum(x*x for x in data) ** 0.5
                normalized_data = [x / scale for x in data]
                norm.append(normalized_data)
            
        return norm

class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        data = np.array(features)
        min = np.amin(data, axis=0)
        dif = np.amax(data, axis=0) - min
        for i in range(len(dif)):
            if dif[i] == 0:
                data[:, i] = min[i]
                dif[i] = 1
            
        return [list((d - min)/ dif) for d in data]
