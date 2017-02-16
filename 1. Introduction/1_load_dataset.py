from sklearn import datasets


iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)      # n_samples, n_features
print(digits.target)    # n_targets
print(digits.images[0])

print iris.data
