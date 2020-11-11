import json
from typing import Dict, Any, Union
import pandas as pd
import re


def read_json_file(json_file: str) -> Dict[Any,Any]:
    with open(json_file, 'r') as file:
        json_dict = json.load(file)
    return json_dict


class RemoveDuplicateCharacter:
    """
    Remove duplicate character 2+ into 2. "aaaa" -> "aa"
    (Assume lowercase)
    """

    def run(self, data: pd.Series, *args, **kwargs) -> Union[Dict[str, object],
                                                             Dict[str, pd.Series]]:
        if 'verbose' in kwargs and kwargs['verbose']:
            print("running duplicate removal character")
        data = data.apply(lambda x : re.sub(r'([a-z])\1{2,}', r'\1\1', x))
        data = data.apply(lambda x : re.sub(r'([?!,.:;\'\">\[\]})(~]){2,}', r'\1', x)) # handle symbol
        return {
            'data' : data
        }

    def prepare_data(self, *args, **kwargs):
        pass
