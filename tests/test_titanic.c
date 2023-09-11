#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef RANDOM_FORREST
#include"random_forrest_classifier.h"
#endif
#ifdef LOGISTIC_REGRESSION
#include"logistic_regression.h"
#endif

void main(){
    #ifdef RANDOM_FORREST
    TS_RandomForestClassifier st_model;
    #endif
    #ifdef LOGISTIC_REGRESSION
    TS_Logistic_Regression st_model;
    #endif
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
        st_model.Pclass = atoi(token);
        token = strtok(NULL, delim);  // SibSp
        st_model.SibSp = atoi(token);
        token = strtok(NULL, delim);  // Parch
        st_model.Parch = atoi(token);
        token = strtok(NULL, delim);  // Fare
        st_model.Fare = atof(token);
        token = strtok(NULL, delim);  // Age
        st_model.Age = atof(token);
        token = strtok(NULL, delim);  // Sex_female        
        st_model.Sex_female = (strcmp(token,"True") == 0)? 1:0;
        token = strtok(NULL, delim);  // Sex_male
        st_model.Sex_male = (strcmp(token,"True") == 0)? 1:0;
        token = strtok(NULL, delim);  // ans
        ans = atoi(token);
        pred = predict(&st_model);
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