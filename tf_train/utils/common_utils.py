import json
import functools
from pathlib import Path
from collections import OrderedDict


def read_json(fname: str) -> dict:
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: str) -> None:
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def rgetattr(obj, attr, *args):
    """
    recursively get attrs. i.e. rgetattr(module, "sub1.sub2.sub3")
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
