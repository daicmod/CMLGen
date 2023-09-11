
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from CMLGen.RFCGen import RFCGen
if __name__ == '__main__':
    train_data = pd.read_csv("./titanic_train.csv")

    y = train_data["Survived"]
    train_data = train_data.fillna({'Age':0, 'Fare':0})

    features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
    X = pd.get_dummies(train_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)

    crfc = RFCGen(model, X.dtypes, ["ANS_NO", "ANS_YES"])
    crfc.write("./")

    pred = model.predict(X)
    X["pred"] = pred
    X.to_csv("./titanic_train_ans.csv", index=False )