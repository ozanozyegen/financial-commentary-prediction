import numpy as np
from math import ceil
from sklearn.utils import resample
from collections import Counter

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))
    
def multivariate_data(dataset, target, start_index, end_index, history_size,
        target_size=1, stride=1, step=1, single_step=False, autoregressive=True):
    """ Train test split of multivariate data for Neural Network training """
    data, labels = [], []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index, stride):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if autoregressive: # DeepAR style
            labels.append(target[i-history_size:i])
        elif single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def undersampler(X, y):
    """ Undersamples the dominant class to the size of the second largest class
    Arguments:
        X:
        y: 1D sparse classification labels
    Returns
        X_downsampled:
        Y_downsampled:
    """
    X, y = np.array(X), np.array(y)
    
    class_counts = Counter(y)
    class_counts_List = sorted(list(class_counts.items()), key=lambda x:x[1])
    
    largest = class_counts_List[-1]
    secondLargest = class_counts_List[-2]
    
    class_instances = {k:[] for k in np.unique(y)}
    for i,r in enumerate(y):
        class_instances[r].append(i)
    
    resampledLargest = resample(class_instances[largest[0]], n_samples = secondLargest[1], random_state=101)
    
    X_downsampled = []
    Y_downsampled = []
    for class_ in class_instances.keys():
        if class_ != largest[0]:
            X_downsampled.extend(X[class_instances[class_]])
            Y_downsampled.extend([class_ for _ in class_instances[class_]])
        else:
            X_downsampled.extend(X[resampledLargest])
            Y_downsampled.extend([class_ for _ in range(secondLargest[1])])
    
    return np.array(X_downsampled), np.array(Y_downsampled)