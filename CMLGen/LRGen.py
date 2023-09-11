import numpy as np
from typing import List
from typing import Dict
from sklearn.linear_model import LogisticRegression
from CMLGen.MLGen import MLGen

class LRGen(MLGen):
    def __init__(self, 
                model:LogisticRegression,                  
                feature_types : Dict[str, str],
                answer_names : List[str],
                ):
        if hasattr(model, 'coef_') and not model.coef_.any():
            ValueError("The model is not fitted. Please fit the model first.")
        if len(model.feature_names_in_) != len(feature_types.keys()):
            ValueError("The required number of features is " + str(len(model.feature_importances_)) + ", but the number of features specified in feature_types is " + str(len(feature_types.keys))  +".")
        if len(model.classes_) != len(answer_names):
            ValueError("The required number of answer is " + str(len(model.classes_)) + ", but the number of answer specified in answer_names is " + str(len(answer_names))  +".")
        super().__init__(feature_types, answer_names)

        self.model = model
        self._generate_header()
        self._generate_code()

        self.cfile = "logistic_regression.c"
        self.hfile = "logistic_regression.h"

    def _generate_header(self):
        self.header = "#ifndef LOGISTIC_REGRESSION_H\n"
        self.header += "#define LOGISTIC_REGRESSION_H\n\n"
        self.header += "typedef enum {\n"
        for ans in self.answer_names:
            self.header += ("\t" + ans + ",\n")
        self.header += ("}E_PRED_ANS;\n\n")

        self.header += "typedef struct {\n"
        for name in self.model.feature_names_in_:
            dtype = str(self.feature_types[name])
            self.header += "\t" + dtype + " " + name + ";\n"
        self.header += ("} TS_Logistic_Regression;\n\n")
        self.header += ("E_PRED_ANS predict(TS_Logistic_Regression* pst_lr);\n\n")
        self.header += "#endif // LOGISTIC_REGRESSION_H"

    def _generate_code(self):
        self.code = '#include"logistic_regression.h"\n\n'
        self._generate_predict_code()

    def _generate_predict_code(self):
        num_regression = len(self.model.classes_)
        self.code += 'E_PRED_ANS predict(TS_Logistic_Regression* pst_lr) {\n'
        self.code += "\tE_PRED_ANS max_ans=0, i;\n"                
        self.code += "\tfloat regression[" + str(num_regression) + "];\n"
        offset = 0
        if num_regression == 2:
            self.code += "\tregression[0] = 0.0f;\n"
            offset = 1
        for i in range(offset, num_regression):
            i2 = i-offset
            self.code += "\tregression[" + str(i) + "] = " + str(self.model.intercept_[i2])
            for j, name in enumerate(self.model.feature_names_in_):
                self.code += " + (pst_lr->" + name + "*" + str(self.model.coef_[i2][j]) + ")"
            self.code += ";\n"
        self.code += '\n'
        self.code += "\tfor(i=0;i<" + str(num_regression) + ";i++)\n"
        self.code += "\t\tmax_ans = (regression[max_ans] < regression[i])?i:max_ans;\n"
        self.code += '\n'
        self.code += "\treturn max_ans;\n"
        self.code += '}\n'