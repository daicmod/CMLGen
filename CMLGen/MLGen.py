import os
from typing import List
from typing import Dict

class MLGen:
    def __init__(self,
                feature_types : Dict[str, str],
                answer_names : List[str],
                ):
        self.feature_types = feature_types
        self.answer_names = answer_names

        # pandasのdtypesを突っ込まれた場合はc言語の型に読み替える
        dtype_mapping = {
            "int64": "long long",
            "int32": "long",
            "int16": "int",
            "int8": "char",
            "bool": "char",
            "float64": "double",
            "float32": "float"
        }
        for k in feature_types.keys():
            dtype = str(self.feature_types[k])
            if dtype in dtype_mapping.keys():
                self.feature_types[k] = dtype_mapping[dtype]
            else:
                pass
    
    def write(self, dst : str):
        with open(os.path.join(dst, self.hfile), 'w') as file:
            file.write(self.header)

        with open(os.path.join(dst, self.cfile), 'w') as file:
            file.write(self.code)
