from loadCIFAR import loadCIFAR10
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# a magic function we provide
# flatten out all images to be one-dimensional
Xtr, Ytr, Xte, Yte = loadCIFAR10()
Xtr_rows = Xtr.reshape(Ytr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Yte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

    # use a particular value of k and evaluation on validation data
    nn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, p=2)
    print "Fitting with K: {} ...\n".format(k)
    print "Predicting with K: {} ...\n".format(k)

    scores = cross_val_score(nn, Xtr_rows, Ytr, cv=5)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

    validation_accuracies.append((k, scores.mean(), scores.std()))
print validation_accuracies

validation_accuracies = np.array(validation_accuracies)

max_acc = -1.0
best_k = 0
for k, acc, _ in validation_accuracies:
    if max_acc < acc:
        best_k = k
        max_acc = acc

clf = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1, p=1)
clf.fit(Xtr_rows, Ytr)
YPr = clf.predict(Xte_rows)

print "Test Accuracy {}" % (np.mean(Yte == YPr))