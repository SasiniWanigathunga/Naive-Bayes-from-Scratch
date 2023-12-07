import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class NaiveBayesClassifier:
    """
    Naive Bayes classifier.

    This class implements a Naive Bayes classifier. It can be used to classify data into one of two or more classes. It assumes that the features are independent of each other, and that the variance of each feature is the same for each class.

    Attributes
    ----------
    X_train : np.ndarray
        The features in the training data. Each row corresponds to a single data point, and each column corresponds to a feature.
    y_train : np.ndarray
        The labels for the training data. Each entry corresponds to the label for the corresponding row in `X_train`.
    classes : np.ndarray
        The unique labels in the training data.
    parameters : dict
        The mean and variance for each feature in each class.
    class_probabilities : dict
        The prior probability for each class.

    Methods
    -------
    fit()
        Fit the model to the training data.
    log_likelihood(X, mean, variance)
        Calculate the log likelihood.
    predict(X_test)
        Predict the labels for the test data.

    Examples
    --------
    >>> X_train = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]])
    >>> y_train = np.array([0, 1, 0])
    >>> model = NaiveBayesClassifier(X_train, y_train)
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.parameters = {}
        self.class_probabilities = {}

    def fit(self) -> None:
        """
        Fit the model to the training data.

        This method calculates the mean, variance, and prior probability for each feature in the training set for each class.

        Examples
        --------
        >>> model.fit()
        """
        for _, c in enumerate(self.classes):
            X_train_c = self.X_train[self.y_train == c]
            self.parameters["mean_" + str(c)] = X_train_c.mean(axis=0)
            self.parameters["var_" + str(c)] = X_train_c.var(axis=0)
            self.class_probabilities[c] = np.sum(
                self.y_train == c) / self.y_train.shape[0]

    def _log_likelihood(self, X: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
        """
        Calculate the log likelihood.

        Parameters
        ----------
        X : np.ndarray
            Data.
        mean : np.ndarray
            Mean of the data.
        variance : np.ndarray
            Variance of the data.

        Returns
        -------
        np.ndarray
            Log likelihood of the data. This is a 1-dimensional array. 

        Examples
        --------
        >>> X = np.array([1, 2, 3])
        >>> mean = np.array([1, 2, 3])
        >>> variance = np.array([1, 1, 1])
        >>> model.log_likelihood(X, mean, variance)
        [-0.91893853 -0.91893853 -0.91893853]
        """
        likelihood = np.exp(-(X-mean)**2/(2*variance)) / \
            np.sqrt(2*np.pi*variance)
        return np.log(likelihood)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the test data.

        Parameters
        ----------
        X_test : np.ndarray
            The test data. Each row corresponds to a single data point, and each column corresponds to a feature.

        Returns
        -------
        np.ndarray
            Predicted labels for the test data. This is a 1-dimensional array. 

        Examples
        --------
        >>> X_test = np.array([[1, 2, 3, 4], [33, 4, 5, 6], [5, 6, 7, 8]])
        >>> model.predict(X_test)
        [0 0 0]
        """
        post_probs = np.zeros((X_test.shape[0], len(self.classes)))

        for c in self.classes:
            mean = self.parameters["mean_" + str(c)]
            var = self.parameters["var_" + str(c)]
            log_likelihoods = self._log_likelihood(X_test, mean, var)
            prior = self.class_probabilities[c]
            post_probs[:, c] = np.sum(log_likelihoods, axis=1) + np.log(prior)

        return np.argmax(post_probs, axis=1)


def main():
    """
    Main function to run the NaiveBayesClassifier.

    - loads the iris dataset
    - splits it into training and testing sets
    - scales the features
    - trains the NaiveBayesClassifier on the training set
    - makes predictions on the test set
    - calculates and prints the accuracy of the model

    Examples
    --------
    >>> __main__()
    Accuracy 0.96
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = NaiveBayesClassifier(X_train, y_train)
    model.fit()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy %.2f" % accuracy)


if __name__ == "__main__":
    main()
