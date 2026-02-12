import importlib

from collections import abc
from itertools import repeat
from typing import Tuple


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to load model.")

    class_name = get_obj_from_str(config["target"])
    if 'params' in config:
        if hasattr(class_name, 'Config'):
            module = class_name(config['params'])
        else:
            class_cfg = config.get('params', dict())
            module = class_name(**class_cfg)

    else:
        module = class_name()

    return module


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


True_set = ('yes', 'true', 't', 'y', '1')
False_set = ('no', 'false', 'f', 'n', '0')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in True_set:
        return True
    elif v.lower() in False_set:
        return False
    elif v.lower() == 'none':
        return None
    else:
        raise ValueError("Boolean value expected.")


def _ntuple(n):
    def parse(x) -> Tuple:
        if isinstance(x, abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple