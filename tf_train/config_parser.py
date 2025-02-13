import os
import os.path as osp
import random
import logging
import argparse
from operator import getitem
from datetime import datetime
from typing import List, Dict, Union, Optional
from functools import partial, reduce

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf, DictConfig
from dotenv import dotenv_values
from easydict import EasyDict as edict
from tf_train.logging import setup_logging_config
from tf_train.model.models_info import model_info_dict
from tf_train.utils.tf_utils import count_samples_in_tfr
from tf_train.utils.common import try_bool, try_null, get_git_revision_hash


class ConfigParser:
    """
    class to parse configuration json file.
    Handles hyperparameters for training, initializations of modules,
    checkpoint saving and logging module.

    Args:
        config: DictConfig object with configurations.
        run_id: Unique Identifier for train & test. Used to save ckpts & training log.
        modification: Additional key-value pairs to override in config.
    """
    def __init__(self,
                 config: DictConfig,
                 run_id: Optional[str] = None,
                 verbose: bool = False,
                 modification: dict = None):
        self.config = config

        # Apply any modifications to the configuration
        if modification:
            # Removes keys that have None as values
            modification = {k:v for k,v in modification.items() if v}
            apply_modifications(self.config, modification)

        # set seed
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # If run_id is None, use timestamp as default run-id
        if run_id is None:
            run_id = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.verbose = verbose
        self.git_hash = get_git_revision_hash()

        # Set directories for saving logs, metrics, and models
        save_root = osp.join(config.save_dir, config.experiment_name, run_id)
        _logs_dir = osp.join(save_root, "logs")
        _metrics_dir = osp.join(save_root, "metrics")
        _models_dir = osp.join(save_root, "models")

        # add tensorboard logging dirs
        if config.trainer.use_tensorboard:
            self.config.trainer.tf_scalar_logs = osp.join(save_root, "tf_logs", "scalars")
            self.config.trainer.tf_prune_logs = osp.join(save_root, "tf_logs", "prune")
            self.config.trainer.tf_image_logs = osp.join(save_root, "tf_logs", "images")

        # Create necessary directories
        os.makedirs(_logs_dir, exist_ok=True)
        os.makedirs(_metrics_dir, exist_ok=True)
        os.makedirs(_models_dir, exist_ok=True)

        # dump custom env vars from .env file to config.json
        custom_env_vars = dotenv_values(".env")
        self.config.os_vars = dict(custom_env_vars)

        # check if num_classes count matches number of classes in class_map_txt_path
        n_cls = config.data.num_classes
        class_map_txt = config.data.class_map_txt_path
        assert osp.exists(class_map_txt), f"{class_map_txt} does not exist"
        with open(class_map_txt, 'r', encoding="utf-8") as fptr:
            n_fcls = len(fptr.readlines())
            assert n_fcls == n_cls, f"num_classes {n_cls} and classes in {class_map_txt} {n_fcls} don't match"

        # count total training and validation samples if they are not explicitely provided
        num_train = config.data.num_train_samples
        if num_train:
            print(f"Note: num_train_samples: {num_train} was provided in config.",
                  "Make sure this is correct since train epoch size depends on this")
        else:
            self.config.data.num_train_samples = count_samples_in_tfr(config.data.train_data_dir)
        num_val = config.data.num_val_samples
        if num_val:
            print(f"Note: num_val_samples: {num_val} was provided in config.",
                  "Make sure this is correct since val epoch size depends on this")
        else:
            self.config.data.num_val_samples = count_samples_in_tfr(
                config.data.val_data_dir)

        # Save the updated config to the save directory
        OmegaConf.save(self.config, osp.join(save_root, "config.yaml"))

        # configure logging module
        setup_logging_config(_logs_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        # set model info obj, config obj and resume chkpt
        self._model = edict(model_info_dict[config["arch"]])
        # assign updated log and save dir after saving config
        self.config.logs_dir = _logs_dir
        self.config.metrics_dir = _metrics_dir
        self.config.models_dir = _models_dir

    @classmethod
    def from_args(cls,
                  args: argparse.Namespace,
                  modification: Optional[dict] = None,
                  add_all_args: bool = True):
        """
        Initialize this class from CLI arguments. Used in train, test.
        Args:
            args: Parsed CLI arguments.
            modification: Key-value pair to override in config.
                          Can have nested structure separated by colons.
                          e.g. ["key1:val1", "key2:sub_key2:val2"]
            add_all_args: Add all args to modification 
                          that are not alr present as top-level keys.
        """
        modification = {} if not modification else modification
        # Add all args to modification from args
        if add_all_args:
            # only check top-level keys
            mod_keys = {k.rsplit(':')[0] for k in modification}
            for arg, value in vars(args).items():
                if arg not in mod_keys:
                    modification[arg] = value
        # Override configuration parameters if args.override is provided
        if args.override:
            parse_and_cast_kv_overrides(args.override, modification)

        # Load configuration from YAML
        config = OmegaConf.load(args.config)
        return cls(config, args.run_id, args.verbose, modification)

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
        module_name = _get_by_path(self, name + ["type"])
        module_args = dict(_get_by_path(self, name + ["args"]))
        assert all(k not in module_args for k in kwargs), 'Overwriting kwargs given in config file is not allowed'
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
        module_name = _get_by_path(self, name + ["type"])
        module_args = dict(_get_by_path(self, name + ["args"]))
        assert all([k not in module_args for k in kwargs]
                   ), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name: str):
        """Access items like ordinary dict."""
        return self.__getattr__(name)

    def __getattr__(self, name):
        """Delegate attribute access to the config object."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def setup_logger(self, name: str, verbosity: int = 2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_levels[verbosity])
        print(
            f"Logger {name} initialized with level: {self.log_levels[verbosity]}")

    # set read-only attributes
    @property
    def model(self):
        return self._model
    
    def __str__(self):
        return OmegaConf.to_yaml(self.config)


def apply_modifications(config: DictConfig, modification: dict) -> None:
    """
    Applies modifications to a nested DictConfig object using colon-separated keys inplace.
    Args:
        config (DictConfig): Original configuration object.
        modification (dict): Dictionary with colon-separated keys representing the hierarchy and values to override.
    """
    for key, value in modification.items():
        path = key.split(":")
        node = config
        for part in path[:-1]:  # Traverse to the parent node
            if part not in node:
                node[part] = {}  # Create nested structure if missing
            node = node[part]
        node[path[-1]] = value


def parse_and_cast_kv_overrides(override: List[str], modification: dict) -> None:
    """
    Parses a list of key-val override strings and casts values to appropriate types inplace.
    Args:
        override (List[str]): List of strings in the format "key:value" or "key:child1:child2:....:val".
                 Bool should be passed as true & None should be passed as null
    """
    for opt in override:
        key, val = opt.rsplit(":", 1)
        # Attempt to cast the value to an appropriate type
        for cast in (int, float, try_bool, try_null):
            try:
                val = cast(val)
                break
            except (ValueError, TypeError):
                continue
        modification[key] = val


def _get_by_path(tree: Dict, keys: Union[str, List[str]]):
    """Access a nested object in tree by a key or sequence of keys."""
    return reduce(getitem, keys, tree)
