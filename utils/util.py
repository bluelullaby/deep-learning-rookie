import json
from pathlib import Path
from collections import OrderedDict


def read_json(fname: str):
    fname = Path(fname)
    with fname.open('rt') as file:
        return json.load(file, object_hook=OrderedDict)
