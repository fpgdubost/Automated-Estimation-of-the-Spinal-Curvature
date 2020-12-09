from keras import backend as K
from numpy import linalg as npla

"""
    y_true are true labels
    y_pred are our predictions
"""
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

# idee : compter le nombre de point bien places par rapport a un threshold
def number_of_good_points(y_true, y_pred, threshold):
    nb = 0
    for i in range(len(y_true)):
        dist = npla.norm(y_true[i] - y_pred[i])
        if dist <= threshold:
            nb += 1
    return nb


# idee : moyenne de la distance des points predits aux point voulus
def mean_distance(y_true, y_pred):
    dist = 0
    for i in range(len(y_true)):
        dist += npla.norm(y_true[i] + y_pred[i])
    dist = dist/float(len(y_true))
    return dist
