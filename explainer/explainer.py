from torch import nn
from typing import Union, List, Tuple
from explainer.llm_explainer_utils.hook import ExplainerHook
from explainer.llm_explainer_utils.parser import LayerParser
from explainer import llm_explainer_methods
from explainer.attribute import DecoderPostprocessor
import torch
from explainer.extra.logging import get_logger

logger = get_logger(__name__)

NEED_MODEL_ALGORITHM = ["Gradient", "Attention"]

def get_device():
    if torch.cuda.is_available():
        logger.info("Using NVIDIA CUDA")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


class Explainer(ExplainerHook, LayerParser):
    """
    A class to explain the model.

    Args:
        model (nn.Module): The model to be explained.
        method (Union[str, List[str]]): The method to be used for explanation.

    Returns:
        Explainer: The explainer object.
    """

    def __init__(self, model, method: Union[str, List[str]]):

        super().__init__(model, "root", [])
        LayerParser.__init__(self)

        if isinstance(method, str):
            self.method = [method]
        elif isinstance(method, List):
            self.method = method

        self.device = get_device()

        # TODO: compute 要放在init

    def _hook_impl(self):
        self.layers.clear()

        for res in self.model.named_modules():
            if isinstance(res, Tuple):
                name, layer = res
            else:
                name, layer = None, res

            if not isinstance(layer, nn.Module):
                continue

            hook = ExplainerHook(layer, name, self._flow)

            level = name.count(".")
            level = 0 if not level else level
            parent_name = ".".join(name.split(".")[:-1]) if level > 0 else None

            if parent_name is not None:
                self._update_layers(
                    layers=self._layers, name=parent_name, children=name
                )
            self._update_layers(
                layers=self._layers,
                name=name,
                layer=layer,
                hook=hook,
                level=level,
                parent=parent_name,
            )

            hook.hook()

    def _unhook_impl(self):
        for layer_name in self.layers.keys():
            self.layers[layer_name]["hook"].unhook()
        self.layers.clear()

    def compute(
        self,
        *args,
        **kwargs,
    ):
        """
        Compute the explanation.

        Args:
            *args:
                Additional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                The explanation result, decode prediction and decode input.

        Example:
            >>> import torch
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> from explainer import Explainer, ModelExplanationTask
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> input_text = "Hello, my dog is cute"
            >>> input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
            >>> method = ["Gradient", "Attention"]
            >>> with Explainer(model=model, method=method) as explain:
            >>>      model.eval()
            >>>      output = model.generate(**input_ids.clone())
            >>>      param = {
            >>>      "tokenizer": tokenizer,
            >>>      "output": output,
            >>>      "task_type": ModelExplanationTask.GENERATIVE_TEXT_CHAT,
            >>>      "input_ids": input_ids
            >>>      "layer_names"=["q", "k", "v"],
            >>>      "positions"=[0, 0, 0],
            >>>      "is_input"=False,
            >>>      "strict"=False)}
            >>>      result = explain.compute(**param)
        """
        kwargs = self.explainer_handler.generate_parameter(**kwargs)
        result = {}
        for method in self.method:
            worker = getattr(llm_explainer_methods, method)()
            kwargs.update(
                {"model": self.model}
            ) if method in NEED_MODEL_ALGORITHM else None
            result[method] = worker.compute(layers=self.layers.copy(), *args, **kwargs)
        output = kwargs.pop("output", None)
        tokenizer = kwargs.pop("tokenizer", None)
        input_ids = kwargs.pop("input_ids", None)
        predict_decode, output_decode, prompt_decode = DecoderPostprocessor(
            output=output, input_ids=input_ids, tokenizer=tokenizer
        )(*args, **kwargs)
        result.update(
            {
                "Prediction": predict_decode,
                "Decode_output": output_decode,
                "Prompt": prompt_decode,
            }
        )
        # Gradient: raw, saliency, gradient_x_input_per_block
        #   saliency.shape -> (N_i)
        #   gradient_x_input_per_block.shape -> (N_i, N_block)
        # Attention: qkv
        return result
