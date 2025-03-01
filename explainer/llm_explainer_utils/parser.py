import re
from torch import nn
from typing import List, Union, Any, Dict

from explainer.llm_explainer_utils.hook import LayerHook


class LayerParser:
    def __init__(self):
        self._layers = dict()

    @property
    def layers(self) -> Dict[str, Dict[str, Any]]:
        return self._layers

    def _update_layers(
        self,
        layers: Dict[str, Dict[str, Any]],
        name: str,
        layer: nn.Module = None,
        hook: LayerHook = None,
        level: int = None,
        parent: str = None,
        children: str = None,
    ) -> None:
        if name not in layers.keys():
            layers[name] = {
                "layer": None,
                "hook": None,
                "level": None,
                "parent": None,
                "children": [],
            }

        if layer:
            layers[name]["layer"] = layer

        if hook:
            layers[name]["hook"] = hook

        if level:
            layers[name]["level"] = level

        if parent:
            layers[name]["parent"] = parent

        if children:
            layers[name]["children"].append(children)

        return


def get_layer(
    layers: Dict[str, Dict[str, Any]], layer_name: Union[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Get layers with strict name or names.

    Args:
        layer_name (str, List[str]): Name of layer or layers.
    Return:
        result
    """
    result = {}
    if isinstance(layer_name, List):
        for name in layer_name:
            if name not in layers.keys():
                raise KeyError('Layer "{name}" not hooked or not existed.')
            result[name] = layers[name]
    else:
        if layer_name not in layers.keys():
            raise KeyError('Layer "{layer_name}" not hooked or not existed.')
        result[layer_name] = layers[layer_name]
    return result


def parse_layer(
    layers: Dict[str, Dict[str, Any]], substring: str
) -> Dict[str, Dict[str, Any]]:
    """
    Get layers with naming pattern.

    Args:
        substring (str): Naming pattern of layers.
    Return:
        result
    """
    result = {}
    pattern = r"\b\w*{}.*?\b".format(re.escape(substring))
    for layer_name in layers.keys():
        matches = re.findall(pattern, layer_name)
        if matches:
            result[layer_name] = layers[layer_name]
    return result
