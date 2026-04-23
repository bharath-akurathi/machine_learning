## 1) Classification Template

```python id="simple1"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
```

(Replace model with Logistic Regression, K-Nearest Neighbors, etc.)

---

## 2) Regression Template

```python id="simple2"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
```

---

## 3) Clustering Template

```python id="simple3"
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("data.csv")

model = KMeans(n_clusters=3)
model.fit(data)

print(model.cluster_centers_)
```

---

### Just remember:

* Output column present + categorical → Classification
* Output column present + numeric → Regression
* No output column → Clustering

