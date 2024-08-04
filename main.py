import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from arpalearn import train_test_split, DecisionTree, LinearRegression

data = pd.read_csv('Salary_dataset.csv')
X, y = data.drop(columns=['Salary', 'index'], axis=1), data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)

model = LinearRegression(epochs=300)
model.fit(X_train, y_train)

print(model.predict(10.4), 122392.0)

plt.scatter(data.YearsExperience, data.Salary)
plt.plot(list(range(1,15)), [model.mCoefficient * x + model.bIntercept for x in range(1, 15)], color="red")
plt.show()

# print(len(X_train), len(X_test), len(y_train), len(y_test))
# print(X_train.head(), X_test.head())
'''

model = DecisionTree()
print(model.check_purity(y_train[X_train.petal_width > 0.8]))
print(model.classify_data(y_train[(X_train.petal_width > 0.8) & X_train.petal_width < 2]))
potential_splits = model.get_potential_splits(X_train.values)

# model.calculate_entropy(X_train)
label_column = X_train.iloc[:, -1]
_, counts = np.unique(label_column, return_counts=True)
print (_, counts)
# print(np.unique(y_train))


visual = X_train
visual['species'] = y_train
# print(visual)
sns.lmplot(visual, x = 'petal_width', y = 'petal_length', hue='species', fit_reg=False)
plt.vlines(x=potential_splits[3], ymin = 1, ymax= 7)
plt.hlines(y=potential_splits[2], xmin= 0, xmax=2.5)
plt.show()
'''