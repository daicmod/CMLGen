
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from CMLGen.LRGen import LRGen
if __name__ == '__main__':
    train_data = pd.read_csv("./titanic_train.csv")

    y = train_data["Survived"]
    train_data = train_data.fillna({'Age':0, 'Fare':0})

    features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
    X = pd.get_dummies(train_data[features])

    model = LogisticRegression()
    model.fit(X, y)

    clr = LRGen(model, X.dtypes, ["ANS_NO", "ANS_YES"])
    clr.write("./")

    pred = model.predict(X)
    X["pred"] = pred
    X.to_csv("./titanic_train_ans.csv", index=False )