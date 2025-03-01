import warnings
import torch
import logging
from typing import List, Union, Any, Tuple, Dict
from explainer.llm_explainer_utils.parser import get_layer, parse_layer
from explainer.attribute import AttentionPostprocessor
from explainer.extra import get_logger

logger = get_logger(__name__)


class Attention:
    def __init__(self, *args, **kwargs):
        self._qkv = {}

    @property
    def qkv(self):
        return self._qkv

    def _update_qkv(self, qkv, component, layer_name, parent, tensor):
        if parent not in qkv.keys():
            qkv[parent] = dict()
        qkv[parent][component] = {}
        qkv[parent][component]["layer_name"] = layer_name
        qkv[parent][component]["tensor"] = tensor

    def _build_qkv(self, qkv, component, is_input, position, layers):
        in_or_out = "input" if is_input else "output"
        for layer_name in layers.keys():
            parent = layers[layer_name]["parent"]
            # TODO: attention和gradient目前只抓第一次執行後的結果
            tensor = getattr(layers[layer_name]["hook"], in_or_out)

            if isinstance(tensor, (Tuple, List, torch.Tensor)):
                if position >= len(tensor):
                    message = f'"position" of "{component}" is out of range.\
                    Module "{layer_name}" has {len(tensor)} {in_or_out} only,\
                    but got position {position}.\
                    Please check the position of {component}!'
                    warnings.warn(message, UserWarning)
                    tensor = None
                    continue
                # TODO:為配合hook.py要做新的更動
                tensor = tensor[position]
            else:
                if position != 0:
                    message = f'\
                    Provided "position" of "{component}" is out of range.\
                    Module "{layer_name}" has 1 {in_or_out} only,\
                    but got position {position}.\
                    Please check the position of {component}!'
                    warnings.warn(message, UserWarning)

            if tensor is not None:
                self._update_qkv(
                    qkv=qkv,
                    component=component,
                    layer_name=layer_name,
                    parent=parent,
                    tensor=tensor,
                )

    def _trim_qkv(self, qkv):
        required = {"q", "k", "v"}
        pop_list = [
            parent
            for parent in qkv.keys()
            if not required.issubset(set(qkv[parent].keys()))
        ]
        for key in pop_list:
            qkv.pop(key)

    def locate_q(
        self,
        layers: Dict[str, Dict[str, Any]],
        layer_name: Union[str, List[str]],
        is_input: bool,
        position: int,
        strict: bool = False,
    ):
        """
        Perform a specific operation on a layer\
        or layers based on the provided criteria.

        Args:
            layer_name (str):
                The name or naming pattern of attributes of self.module where queries are located.
                Multiple names/patterns are supported.
            is_input (bool):
                Specifies whether queries represent the input of the given layer.
                If True, queries are considered input; if False, they represent outputs.
            position (int):
                The position of the query in the input/output arguments. Should be an integer.
                If there's only one argument, position is 0.
            strict (bool):
                If True, the function will search for a layer with an exact name matching 'layer_name'.
                If False, it will find layers with naming patterns matching 'layer_name'.
                Defaults to False.
        Returns:
            None
        """
        if strict:
            layers = get_layer(layers, layer_name)
        else:
            layers = parse_layer(layers, layer_name)

        self._build_qkv(
            qkv=self.qkv,
            component="q",
            is_input=is_input,
            position=position,
            layers=layers,
        )

    def locate_k(
        self,
        layers: Dict[str, Dict[str, Any]],
        layer_name: Union[str, List[str]],
        is_input: bool,
        position: int,
        strict: bool = False,
    ):
        """
        Perform a specific operation on a layer or layers based on the provided criteria.

        Args:
            layer_name (str): The name or naming pattern of attributes of self.module where keys are located.
                            Multiple names/patterns are supported.
            is_input (bool): Specifies whether keys represent the input of the given layer.
                            If True, keys are considered input; if False, they represent outputs.
            position (int): The position of the key in the input/output arguments. Should be an integer.
                            If there's only one argument, position is 0.
            strict (bool): If True, the function will search for a layer with an exact name matching 'layer_name'.
                        If False, it will find layers with naming patterns matching 'layer_name'.
                        Defaults to False.
        Returns:
            None
        """
        if strict:
            layers = get_layer(layers, layer_name)
        else:
            layers = parse_layer(layers, layer_name)

        self._build_qkv(
            qkv=self.qkv,
            component="k",
            is_input=is_input,
            position=position,
            layers=layers,
        )

    def locate_v(
        self,
        layers: Dict[str, Dict[str, Any]],
        layer_name: Union[str, List[str]],
        is_input: bool,
        position: int,
        strict: bool = False,
    ):
        """
        Perform a specific operation on a layer or layers based on the provided criteria.

        Args:
            layer_name (str): The name or naming pattern of attributes of self.module where values are located.
                            Multiple names/patterns are supported.
            is_input (bool): Specifies whether values represent the input of the given layer.
                            If True, values are considered input; if False, they represent outputs.
            position (int): The position of the value in the input/output arguments. Should be an integer.
                            If there's only one argument, position is 0.
            strict (bool): If True, the function will search for a layer with an exact name matching 'layer_name'.
                        If False, it will find layers with naming patterns matching 'layer_name'.
                        Defaults to False.
        Returns:
            None
        """
        if strict:
            layers = get_layer(layers, layer_name)
        else:
            layers = parse_layer(layers, layer_name)

        self._build_qkv(
            qkv=self.qkv,
            component="v",
            is_input=is_input,
            position=position,
            layers=layers,
        )

    def cal_attn(self, q, k, v):
        try:
            scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(
                torch.tensor(k.size(-1)).float()
            )
        except ZeroDivisionError:  # ZeroDivisionError
            logging.error(
                "ZeroDivisionError: denominator is zero in Attention.cal_attn."
            )
            scores = torch.zeros_like(q)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        return weights, out

    def compute(
        self,
        layers: Dict[str, Dict[str, Any]],
        layer_names: List[str],
        positions: List[int],
        is_input: bool,
        strict: bool = False,
        model: torch.nn.Module = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            layer_names (List):
                The naming pattern of 'q', 'k', 'v' module of the model.
                Should be in the order of [q's pattern, k's pattern, v's pattern].
            positions (List):
                The position of argument q, k, v in the given module.
                Should be in the order of [q's position, k's position, v's position].
            is_input (bool):
                Is the q, k, and v input or output of the given module?
                If True, (q, k, v) are extracted from the input of the given module.
                Else, (q, k, v) are extracted from the output of the given module.
            strict (bool):
                Is the given 'layer_names' exact as the name of module?
                If True, only modules with names exactly the same as 'layer_names' are considered.
                Else, modules with names partially matched the 'layer_names' are considered.

        Returns:
            dict:
                A dictionary containing the computed attention weights and output of the model.

        Raises:
            RuntimeError:
                If 'self.qkv' is empty, no valid layer names are provided for query, key, value
                layers to build 'self.qkv'.

        Example:
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> from explainer import Explainer, ModelExplanationTask
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> input_text = "Hello, my dog is cute"
            >>> input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
            >>> with Explainer(module=model, method="Attention") as explain:
            >>>     model.eval()
            >>>     output = model.generate(**input_ids.clone())
            >>>     param = {
            >>>         "tokenizer": tokenizer,
            >>>         "output": output,
            >>>         "task_type": ModelExplanationTask.GENERATIVE_TEXT_CHAT,
            >>>         "input_ids": input_ids
            >>>         "layer_names"=["q", "k", "v"],
            >>>         "positions"=[0, 0, 0],
            >>>         "is_input"=False,
            >>>         "strict"=False)}
            >>>     result = explain.compute(**param)
        """
        self._qkv.clear()

        self.locate_q(
            layers=layers,
            layer_name=layer_names[0],
            position=positions[0],
            is_input=is_input,
            strict=strict,
        )
        self.locate_k(
            layers=layers,
            layer_name=layer_names[1],
            position=positions[1],
            is_input=is_input,
            strict=strict,
        )
        self.locate_v(
            layers=layers,
            layer_name=layer_names[2],
            position=positions[2],
            is_input=is_input,
            strict=strict,
        )

        self._trim_qkv(self._qkv)
        if len(self._qkv) == 0:
            raise RuntimeError(
                '"self._qkv" is empty. Please provide valid layer names for query, key, valye\
                layers to build "self.qkv.'
            )

        # TODO: encounter problem(q, k, v shape not same) in Encoder-Decoder model
        for parent in self._qkv.keys():
            try:
                q = self._qkv[parent]["q"]["tensor"].squeeze(0)
                k = self._qkv[parent]["k"]["tensor"].squeeze(0)
                v = self._qkv[parent]["v"]["tensor"].squeeze(0)
                weights, out = self.cal_attn(q, k, v)
                sparse = weights.to_sparse()
                if sparse.is_cuda:
                    sparse = sparse.cpu()
                indices = sparse.indices().T.detach().numpy().tolist()
                values = sparse.values().detach().numpy().tolist()
                # self._qkv[parent]["indices"] = indices
                # self._qkv[parent]["values"] = values
                self._qkv[parent]["pair_indices_values"] = {}
                for i in range(len(indices)):
                    self._qkv[parent]["pair_indices_values"].update(
                        {str(indices[i]): values[i]}
                    )
            except Exception as e:
                logger.error(f"Error in Attention.compute: {e}")
                self._qkv[parent]["pair_indices_values"] = {}

        # TODO: attention和gradient目前只抓第一次執行後的結果，
        # 導致抓Output的話除非抓每次loop的結果，不然只有一個輸出

        # Safeguard : use self._qkv.copy() as input of Postprocessor.median_attention_layers
        # to avoid changing self._qkv

        self._qkv["pair_result"] = AttentionPostprocessor.median_attention_layers(
            self._qkv.copy()
        )

        attention_function = kwargs.get("attention_function", None)

        if attention_function is not None:
            self._qkv[
                "attention_result"
            ] = AttentionPostprocessor().median_attention_result(
                attention_dict=attention_function()
            )
            # TODO: query, keys, values not postprocess yet
            logger.info(
                """The attention result relationship between output
                  and input is available. The toolbox has calculated
                  the relationship."""
            )
        else:
            logger.info(
                """The attention result relationship between output
                  and input is unavailable because the current model
                  lacks rotary position embeddings. Therefore, the
                  toolbox cannot calculate this relationship. Skip
                  the relationship processing."""
            )
        # import json
        # json.loads([*result["Attention"]['pair_result'].keys()][0])
        return self._qkv.copy()
