from tf_train.utils import read_json, write_json, rgetattr
import json
from collections import OrderedDict


JSON_CONTENT = {"a": 1, "b": "c", "d": {"e": 2}}


def test_read_json(tmp_path):
    fname = tmp_path / "test.json"
    with fname.open('wt') as handle:
        json.dump(JSON_CONTENT, handle, indent=4, sort_keys=False)

    assert read_json(fname) == JSON_CONTENT


def test_write_json(tmp_path):
    fname = tmp_path / "test.json"
    write_json(JSON_CONTENT, fname)
    with fname.open('rt') as handle:
        read_content = json.load(handle, object_hook=OrderedDict)

    assert read_content == JSON_CONTENT


def test_recursive_get_attrb():
    class bar:
        class foo:
            class bar1:
                class foo2:
                    sample: int = 1
    x = bar()
    assert rgetattr(x, "foo.bar1.foo2.sample") == 1
