import numpy as np
import random

def train_test_split(X, y, test_size, random_state = None):
    '''
    This method is used to split training and test data.

    Defaults:
    ---------
    random_state = None
    
    random_state is used to set seed.
    No seed is set if it is not provided by the user.

    test_size:
    ----------
    can be float or int
    if int: gets test sample size of int value provided.
        Example: if user provides 10 -> randomly picks 10 test sample from input.
    if float: that is if user provides percentage. Percentage of test sample is fetched.
        Example: if user provides 0.1 -> 10 percent of input is fetched randomly for test data.


    Returns:
    --------
    X_train, X_test, y_train, y_test

    '''
    random.seed(random_state)
    if isinstance(test_size, float):
        test_size = round(test_size * len(X))

    indices = list(range(len(X)))
    test_indices = random.sample(indices, test_size)

    X_train = X.drop(test_indices)
    X_test = X.loc[test_indices]
    y_train = y.drop(test_indices) 
    y_test = y.loc[test_indices]
    
    return X_train, X_test, y_train, y_test


class LinearRegression:
    '''
    This class uses gradient descent to solve Linear Regression

    Attributes:
    -----------
    mCoefficient
    bIntercept
    learningRate
    epochs


    Defaults: 
    ---------
    Learning rate: L: float = 0.001
    Epochs: epochs: int = 300

    fit(X, y):
    ----------
    this method fits the model

    predict(X):
    -----------
    this method predict the model
    '''
    def __init__(self, L = 0.001, epochs = 300) -> None:
        self.mCoefficient = 0
        self.bIntercept = 0
        self.learningRate = L
        self.epochs = epochs

    def gradientDescent(self, X, y, mNow, bNow, L):
        '''
        This method uses partial derivative of loss or cost function of Mean Square Error (MSE) 
        to solve for coefficient (m) and intercept (b). (y = mx + b)
        '''
        mGradient, bGradient = 0,0
        n = len(X)

        for i in range(n):
            curIndependent = X.iloc[i]
            curDependent = y.iloc[i]
            mGradient += -(2/n) * curIndependent * (curDependent - (mNow*curIndependent + bNow))
            bGradient += -(2/n) * (curDependent - (mNow*curIndependent + bNow))

        self.mCoefficient = mNow - L * mGradient.iloc[0]
        self.bIntercept = bNow - L * bGradient.iloc[0]

    def fit(self, X, y):
        m, b = 0, 0 
        
        for i in range(self.epochs):
            if i % 50 == 0:
                print(f"epochs: {i}")
            self.gradientDescent(X, y, m, b, self.learningRate)
            m = self.mCoefficient
            b = self.bIntercept


    def predict(self, X):
        #element-wise calculation
        return self.mCoefficient * X + self.bIntercept


class DecisionTree:
    # def __init__(self, data) -> None:
    #     self.data = data

    #data pure check
    def check_purity(self, label) -> bool:
        npUnique = np.unique(label)
        print(npUnique)
        if len(npUnique) == 1:
            return True
        else: return False

    #classify
    def classify_data(self, label):
        uniqueClasses, countUniqueClasses = np.unique(label, return_counts=True)
        index = countUniqueClasses.argmax()
        classification = uniqueClasses[index]
        return classification
    
    #potential splits
    def get_potential_splits(self, data):
        potential_splits = {}
        for column_index in range(data.shape[1]):
            potential_splits[column_index] = []
            values = data[:, column_index]
            uniqueValues = np.unique(values)

            for row_index in range(1, len(uniqueValues)):
                curValue = uniqueValues[row_index]
                prevValue = uniqueValues[row_index - 1]
                middleValue = (curValue + prevValue ) / 2
                potential_splits[column_index].append(middleValue)  

        return potential_splits

    def split_data(self, data, feature, value):
        dataBelow = data[data[:, feature] <= value]
        dataAbove = data[data[:, feature] > value]
        return dataBelow, dataAbove
    
    #Lowest Overall Entropy
    def calculate_entropy(self, data):
        label_column = data[:, -1]
        _, counts = np.unique(data, return_counts=True)
        print (counts)
        # entropy = 0

        # return entropy