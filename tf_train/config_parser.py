import logging
from pathlib import Path
from datetime import datetime
from functools import partial

from dotenv import dotenv_values
from easydict import EasyDict as edict
from tf_train.logging import setup_logging
from tf_train.model.models_info import model_info_dict
from tf_train.utils.common_utils import read_json, write_json


class ConfigParser:
    def __init__(self, config: dict, resume: str = None, run_id: str = None):
        """
        class to parse configuration json file.
        Handles hyperparameters for training, initializations of modules,
        checkpoint saving and logging module.
        :param config: Dict with configs & HPs to train. contents of `config/train_image_clsf.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param run_id: Unique Identifier for training processes. Used to save ckpts & training log. Timestamp is used as default
        """
        # set save_dir where trained model and log will be saved.
        save_dir = Path(config['trainer']['save_dir'])
        exper_name = config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%Y%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'logs' / exper_name / run_id

        # add tensorboard logging dirs
        if config["trainer"]["use_tensorboard"]:
            _slog = str(save_dir / "tf_logs" / run_id / "scalars")
            _plog = str(save_dir / "tf_logs" / run_id / "prune")
            _ilog = str(save_dir / "tf_logs" / run_id / "images")
            config["trainer"]["tf_scalar_logs"] = _slog
            config["trainer"]["tf_prune_logs"] = _plog
            config["trainer"]["tf_image_logs"] = _ilog

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # dump custom env vars from .env file to config.json
        custom_env_vars = dotenv_values(".env")
        config["os_vars"] = custom_env_vars

        # save updated config file to the checkpoint dir
        write_json(config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        # set model info obj, config obj and resume chkpt
        self._model = edict(model_info_dict[config["arch"]])
        self._config = config
        self.resume = resume

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        return cls(config, resume)

    def _get_val_from_key_list(self, key_list, key_name):
        """
        Recursively get `key_name` from config dict obj
        using `key_list` as an ordered seq of key access.
        Raise IndexError if key_name is absent in dict or
        raise ValueError if key_list contains keys missing in config or non-dict objs

        config._get_dict_key_val(['a', 'b'], 'foo') == conifg['a']['b']['foo']
        if config has a sub-dict element like {'a': {'b': 'foo': 'bar'}, ...}
        """
        def _recursive_get(dct, idx):
            if idx == len(key_list):
                raise IndexError(
                    f"Passed key_list {key_list} is missing key ``{key_name}``")
            obj = dct[key_list[idx]]
            if not isinstance(obj, dict):
                raise ValueError(f"{key_list[idx]} is not a dict obj")
            if key_name in obj:
                return obj[key_name]
            return _recursive_get(obj, idx + 1)
        return _recursive_get(self, 0)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a object handle with the 'name' given as 'type' in config, and
        returns the instance initialized with corresponding arguments given.
        'name' can also be list of keys and subkeys following access order in list
        to get the final 'type' and 'args' keys.

        `object = config.init_obj('name', module, a, b=1)` == `object = module.name(a, b=1)`
        `object = config.init_obj(['name', 'subname'], module, a, b=1)` == `object = module.name.subname(a, b=1)`
        """
        if isinstance(name, str):
            name = [name]
        module_name = self._get_val_from_key_list(name, key_name="type")
        module_args = dict(self._get_val_from_key_list(name, key_name="args"))
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        'name' can also be list of keys and subkeys following access order in list
        to get the final 'type' and 'args' keys.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        if isinstance(name, str):
            name = [name]
        module_name = self._get_val_from_key_list(name, key_name="type")
        module_args = dict(self._get_val_from_key_list(name, key_name="args"))
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # set read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
