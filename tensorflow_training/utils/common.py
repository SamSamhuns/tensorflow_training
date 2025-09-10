from collections import OrderedDict
from typing import Any, Optional
from pathlib import Path
import subprocess
import functools
import json
from omegaconf import DictConfig


def read_json(fpath: str) -> dict:
    with open(fpath, 'r', encoding="utf-8") as handle:
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


def get_git_revision_hash() -> str:
    """Get the git hash of the current commit. Returns None if run from a non-git init repo"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except subprocess.CalledProcessError as excep:
        print(excep, "Couldn't get git hash of the current repo. Returning None")
    return None


############################ conversion utils ############################


def try_bool(val: Any) -> Optional[bool]:
    """
    Check if val is boolabe (true/false) & converts to bool else raise ValueError
    """
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    raise ValueError(
        f"{val} is not a boolean string. Must be 'true' or 'false' converted in lowecase.")


def try_null(val: Any) -> Optional[None]:
    """
    Check if val is nullable (null/none) & converts to None else raise ValueError
    """
    if val.lower() in {"null", "none"}:
        return None
    raise ValueError(
        f"{val} is not a nullable string. Must be 'null' or 'none'  converted in lowecase.")


def can_be_conv_to_float(var: Any) -> bool:
    """
    Checks if a var can be converted to a float & return bool
    """
    try:
        float(var)
        return True
    except ValueError:
        return False
