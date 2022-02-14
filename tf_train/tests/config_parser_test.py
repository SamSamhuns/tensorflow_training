from tf_train.config_parser import ConfigParser
from tf_train.utils import read_json


def test_config_parser_setup() -> None:
    """
    Note: not a unittest
    """
    config = read_json("config/train_image_clsf.json")
    resume = None
    run_id = "pytest_test"
    config = ConfigParser(config, resume, run_id)
