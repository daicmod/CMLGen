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