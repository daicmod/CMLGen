import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List
from typing import Dict


class RFCGen:
    def __init__(self, model :RandomForestClassifier,
                 feature_names : List[str],
                 feature_types : Dict[str, str],
                 answer_names : List[str],
                 ):
        if hasattr(model, 'n_estimators_') and model.n_estimators_ == 0:
            ValueError("The model is not fitted. Please fit the model first.")
        if len(model.feature_importances_) != len(feature_names):
            ValueError("The required number of features is " + str(len(model.feature_importances_)) + ", but the number of features specified in feature_names is " + str(len(feature_names))  +".")
        if len(model.feature_importances_) != len(feature_types.keys()):
            ValueError("The required number of features is " + str(len(model.feature_importances_)) + ", but the number of features specified in feature_types is " + str(len(feature_types.keys))  +".")
        if len(model.classes_) != len(answer_names):
            ValueError("The required number of answer is " + str(len(model.classes_)) + ", but the number of answer specified in answer_names is " + str(len(answer_names))  +".")
        
        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.answer_names = answer_names
        self.estimators = []
        self.predict = ""
        self.header = ""
        self._make_estimators()
        self._make_predict()
        self._make_header()

    def write(self, dst : str):
        cfile = "random_forrest_classifier.c"
        hfile = "random_forrest_classifier.h"
        with open(os.path.join(dst, hfile), 'w') as file:
            file.write(self.header)

        with open(os.path.join(dst, cfile), 'w') as file:
            file.write('#include"' + hfile + '"\n\n')
            file.write(self.predict)
            for estimator in self.estimators:
                file.write(estimator)

    def _make_header(self):
        self.header += "#ifndef RANDOM_FORREST_CLASSIFIER_H\n"
        self.header += "#define RANDOM_FORREST_CLASSIFIER_H\n\n"
        self.header += "typedef enum {\n"
        for ans in self.answer_names:
            self.header += ("\t" + ans + ",\n")
        self.header += ("}E_PRED_ANS;\n\n")

        self.header += ("typedef struct {\n")
        for i, _ in enumerate(self.feature_names):
            if self.feature_types[i] == "int64":
                self.header += ("\t" + "long long " + self.feature_names[i] + ";\n")
            elif self.feature_types[i] == "int32":
                self.header += ("\t" + "long " + self.feature_names[i] + ";\n")
            elif self.feature_types[i] == "int16":
                self.header += ("\t" + "int " + self.feature_names[i] + ";\n")
            elif self.feature_types[i] == "int8":
                self.header += ("\t" + "char " + self.feature_names[i] + ";\n")
            elif self.feature_types[i] == "bool":
                self.header += ("\t" + "char " + self.feature_names[i] + ";\n")
            elif self.feature_types[i] == "float64":
                self.header += ("\t" + "double " + self.feature_names[i] + ";\n")
            elif self.feature_types[i] == "float32":
                self.header += ("\t" + "float " + self.feature_names[i] + ";\n")
            else:
                pass
        self.header += ("} TS_RandomForestClassifier;\n\n")
        self.header += ("E_PRED_ANS predict(TS_RandomForestClassifier* pst_rf);\n\n")
        self.header += "#endif // RANDOM_FORREST_CLASSIFIER_H"
    
    def _make_predict(self):
        for i, _ in enumerate(self.estimators):
            self.predict += "static void estimator" + str(i) + "(TS_RandomForestClassifier* pst_rf, unsigned long answer_score[]);\n"
        self.predict += "\n"

        self.predict += "E_PRED_ANS predict(TS_RandomForestClassifier* pst_rf){\n"
        self.predict += "\tunsigned long answer_score[" + str(len(self.model.classes_)) + "] = { 0 };\n"
        self.predict += "\tE_PRED_ANS max_count_ans=0, i;\n"

        for i, _ in enumerate(self.estimators):
            self.predict += "\testimator" + str(i) + "(pst_rf, answer_score);\n"
        self.predict += "\n"
        
        self.predict += "\tfor(i=" + self.answer_names[0] + ";i<=" + self.answer_names[-1] +";i++)\n"
        self.predict += "\t\tmax_count_ans = (answer_score[max_count_ans]<answer_score[i])?i:max_count_ans; \n"
        self.predict += "\treturn max_count_ans;\n"
        self.predict += "}\n\n"

    def _make_estimators(self, ):
        for i, estimator in enumerate(self.model.estimators_):
            self.estimators.append("static void estimator" + str(i) + "(TS_RandomForestClassifier* pst_rf, unsigned long answer_score[]){\n")
            self._search_estimator(estimator.tree_, 0, 1)
            self.estimators[-1] += ("}\n\n")

    def _search_estimator(self, tree, i:int, tab:int):
        tabs = "\t" * tab
        if tree.feature[i] == -2:
            return np.argmax(tree.value[i][0])
        
        fname =  self.feature_names[tree.feature[i]]
        threshold = tree.threshold[i]
        dtype = self.feature_types[fname]
        if "float" not in str(dtype):
            threshold = int(threshold)
        else:
            threshold = round(threshold, 4)

        if str(dtype) != "bool":
            self.estimators[-1] += (tabs + "if (pst_rf->" + fname + "<=" + str(threshold) + ") {\n")
        else:
            self.estimators[-1] += (tabs + "if (pst_rf->" + fname + "==" + str(threshold) + ") {\n")
            
        ret = self._search_estimator(tree, tree.children_left[i],tab+1)
        ind = tree.children_left[i]
        if ret != -1:
            total = sum(tree.value[ind][0])
            for j, n in enumerate(self.answer_names):
                proba = int(round(100*tree.value[ind][0][j]/total,0))
                self.estimators[-1] += (tabs + "\tanswer_score[" + n + "]\t+=" + str(proba) + ";\n")
        self.estimators[-1] += (tabs + "} else {\n")
        ret = self._search_estimator(tree, tree.children_right[i],tab+1)
        ind = tree.children_right[i]
        if ret != -1:
            total = sum(tree.value[ind][0])
            for j, n in enumerate(self.answer_names):
                proba = int(round(100*tree.value[ind][0][j]/total,0))
                self.estimators[-1] += (tabs + "\tanswer_score[" + n + "]\t+=" + str(proba) + ";\n")
        self.estimators[-1] += (tabs + "}\n")
        return -1
    
