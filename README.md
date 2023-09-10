# CMLGen
Generate C code from scikit-learn models.

## Introduction

This Python module allows you to easily convert trained scikit-learn RandomForestClassifier models into C code, enabling seamless integration of machine learning models into C/C++ applications.

Whether you want to deploy your scikit-learn models on resource-constrained devices or embed them into existing C/C++ projects, this tool simplifies the process by generating efficient C code.

## Usage
To convert a scikit-learn RandomForestClassifier model to C code, follow these steps:

1. Import the RFCGen class from this module.
1. Instantiate the class with your trained RandomForestClassifier model.
1. Call the write method to generate the C code.
1. include random_forrest_classifier.c/h your project.

Here's an example:

```python
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./titanic_train.csv")

y = train_data["Survived"]
train_data = train_data.fillna({'Age':0, 'Fare':0})

features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
X = pd.get_dummies(train_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

crfc = RFCGen(model, X.columns, X.dtypes, ["ANS_NO", "ANS_YES"])
crfc.write("./")
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include"random_forrest_classifier.h"


void main(){
    TS_RandomForestClassifier st_rfc;
    char line[1024];  // CSVファイルの各行を格納するバッファ
    int first = 1;
    int fail = 0;
    FILE *file  = fopen("titanic_train_ans.csv", "rt");
    if (!file ) {
        // 開くのに失敗したらエラーを出力する
        perror("fopen");
        return ;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        int pred;
        int ans;
        char buff[1024];
        char *token;
        char *delim = ",";
        if (first) {
            first = 0;
            continue;
        }
        strcpy(buff, line);

        // 行をカンマで分割
        token = strtok(line, delim); // Pclass
        st_rfc.Pclass = atoi(token);
        token = strtok(NULL, delim);  // SibSp
        st_rfc.SibSp = atoi(token);
        token = strtok(NULL, delim);  // Parch
        st_rfc.Parch = atoi(token);
        token = strtok(NULL, delim);  // Fare
        st_rfc.Fare = atof(token);
        token = strtok(NULL, delim);  // Age
        st_rfc.Age = atof(token);
        token = strtok(NULL, delim);  // Sex_female        
        st_rfc.Sex_female = (strcmp(token,"True") == 0)? 1:0;
        token = strtok(NULL, delim);  // Sex_male
        st_rfc.Sex_male = (strcmp(token,"True") == 0)? 1:0;
        token = strtok(NULL, delim);  // ans
        ans = atoi(token);
        pred = predict(&st_rfc);
        if (pred!=ans){
            printf("CPRED:%d, PYPRED:%d\n LINE:%s\n", pred, ans, buff);
            fail = 1;
        }
    }
    fclose(file );
    if (fail == 0){
        printf("SUCCESS!\n");
    }
}
```

## test data

Titanic - Machine Learning from Disaster
https://www.kaggle.com/competitions/titanic/data


## License

This project is licensed under the MIT License - see the LICENSE file for details.