import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt


mushroom = fetch_ucirepo(id=73)
X_df = mushroom.data.features.fillna("?")
y_df = mushroom.data.targets

encoder = OrdinalEncoder()
X = encoder.fit_transform(X_df.values).astype(int)
y = (y_df.iloc[:,0].values == "p").astype(int)

n_cats = [len(c) for c in encoder.categories_]

alphas = 2.0 ** np.arange(-15, 6)

train_size = 0.01

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))


aucs = []
accs = []
f1s = []
classifiers = []

for alpha in alphas:

    classifier = CategoricalNB(alpha=alpha, min_categories=n_cats)

    print("Training classifier with alpha=2^%d..." % np.log2(alpha),end="")
    classifier.fit(X_train, y_train)
    print(" done")

    y_pred = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)

    accs.append(accuracy_score(y_test,y_pred))
    f1s.append(f1_score(y_test,y_pred))
    aucs.append(roc_auc_score(y_test,probs[:,1]))
    classifiers.append(classifier)

    print("  Accuracy:",accs[-1]," F1:",f1s[-1]," AUC:",aucs[-1])


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log2(alphas),aucs,label="AUC")
ax.plot(np.log2(alphas),accs,label="Accuracy")
ax.plot(np.log2(alphas),f1s,label="F1")
ax.set_xlabel('log2(alpha)')
ax.set_title('Training Size: 0.01')
ax.legend()
fig.savefig("performance_2.pdf", bbox_inches="tight")


best = np.argmax(aucs)
best_classifier = classifiers[best]
print("Best alpha = 2^%d with AUC = %f" % (np.log2(alphas[best]),aucs[best]))

print("Class priors (edible, poisonous):",np.exp(best_classifier.class_log_prior_))