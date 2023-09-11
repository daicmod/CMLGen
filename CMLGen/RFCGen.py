import numpy as np
from typing import List
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from CMLGen.MLGen import MLGen

class RFCGen(MLGen):
    def __init__(self, 
                model :RandomForestClassifier,
                feature_types : Dict[str, str],
                answer_names : List[str],
                ):
        if hasattr(model, 'n_estimators_') and model.n_estimators_ == 0:
            ValueError("The model is not fitted. Please fit the model first.")
        if len(model.feature_names_in_) != len(feature_types.keys()):
            ValueError("The required number of features is " + str(len(model.feature_importances_)) + ", but the number of features specified in feature_types is " + str(len(feature_types.keys))  +".")
        if len(model.classes_) != len(answer_names):
            ValueError("The required number of answer is " + str(len(model.classes_)) + ", but the number of answer specified in answer_names is " + str(len(answer_names))  +".")
        super().__init__(feature_types, answer_names)

        self.model = model
        self._generate_header()
        self._generate_code()

        self.cfile = "random_forrest_classifier.c"
        self.hfile = "random_forrest_classifier.h"

    def _generate_header(self):
        self.header = "#ifndef RANDOM_FORREST_CLASSIFIER_H\n"
        self.header += "#define RANDOM_FORREST_CLASSIFIER_H\n\n"
        self.header += "typedef enum {\n"
        for ans in self.answer_names:
            self.header += ("\t" + ans + ",\n")
        self.header += ("}E_PRED_ANS;\n\n")

        self.header += "typedef struct {\n"
        for name in self.model.feature_names_in_:
            dtype = str(self.feature_types[name])
            self.header += "\t" + dtype + " " + name + ";\n"
        self.header += ("} TS_RandomForestClassifier;\n\n")
        self.header += ("E_PRED_ANS predict(TS_RandomForestClassifier*);\n\n")
        self.header += "#endif // RANDOM_FORREST_CLASSIFIER_H"

    def _generate_code(self):
        self.code = '#include"random_forrest_classifier.h"\n\n'
        for i, _ in enumerate(self.model.estimators_):
            self.code += "static void estimator" + str(i) + "(TS_RandomForestClassifier* pst_rf, unsigned long answer_score[]);\n"            
        self.code += "\n"

        self._generate_predict_code()
        self._generate_estimators_code()

    def _generate_predict_code(self):
        self.code += "E_PRED_ANS predict(TS_RandomForestClassifier* pst_rf){\n"
        self.code += "\tunsigned long answer_score[" + str(len(self.model.classes_)) + "] = { 0 };\n"
        self.code += "\tE_PRED_ANS max_count_ans=0, i;\n"
        for i, _ in enumerate(self.model.estimators_):
            self.code += "\testimator" + str(i) + "(pst_rf, answer_score);\n"
        self.code += "\n"
        self.code += "\tfor(i=" + self.answer_names[0] + ";i<=" + self.answer_names[-1] +";i++)\n"
        self.code += "\t\tmax_count_ans = (answer_score[max_count_ans]<answer_score[i])?i:max_count_ans; \n"
        self.code += "\treturn max_count_ans;\n"
        self.code += "}\n\n"

    def _generate_estimators_code(self):
        for i, estimator in enumerate(self.model.estimators_):
            self.code += "static void estimator" + str(i) + "(TS_RandomForestClassifier* pst_rf, unsigned long answer_score[]){\n"
            self._generate_estimator_code(estimator.tree_, 0, 1)
            self.code += "}\n\n"

    def _generate_estimator_code(self, tree, i:int, tab:int):
        tabs = "\t" * tab
        if tree.feature[i] == -2:
            return np.argmax(tree.value[i][0])

        fname =  self.model.feature_names_in_[tree.feature[i]]
        threshold = tree.threshold[i]
        dtype = self.feature_types[fname]
        if "float" in str(dtype):
            threshold = round(threshold, 4)
        elif "double" in str(dtype):
            threshold = round(threshold, 4)
        else:
            threshold = int(threshold)

        self.code += (tabs + "if (pst_rf->" + fname + "<=" + str(threshold) + ") {\n")
            
        ret = self._generate_estimator_code(tree, tree.children_left[i],tab+1)
        ind = tree.children_left[i]
        if ret != -1:
            total = sum(tree.value[ind][0])
            for j, n in enumerate(self.answer_names):
                proba = int(round(100*tree.value[ind][0][j]/total,0))
                self.code += (tabs + "\tanswer_score[" + n + "]\t+=" + str(proba) + ";\n")
        self.code += (tabs + "} else {\n")
        ret = self._generate_estimator_code(tree, tree.children_right[i],tab+1)
        ind = tree.children_right[i]
        if ret != -1:
            total = sum(tree.value[ind][0])
            for j, n in enumerate(self.answer_names):
                proba = int(round(100*tree.value[ind][0][j]/total,0))
                self.code += (tabs + "\tanswer_score[" + n + "]\t+=" + str(proba) + ";\n")
        self.code += (tabs + "}\n")
        return -1
