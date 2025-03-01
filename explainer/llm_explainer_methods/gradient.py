import torch
from explainer.attribute import Postprocessor, LossFunctionCalculate
from typing import List, Dict, Any


class Gradient:
    """
    Gradient class is used to compute the gradient of the model.
    """

    def __init__(self, *args, **kwargs):
        self.result = {"result": None, "gradient": None}
        self.gradients = {}

    def compute(
        self, layers: Dict[str, Dict[str, Any]], model: torch.nn.Module, *args, **kwargs
    ):
        """
        Computes the gradients for specified layers.

        Args:
            layers (dict):
                A dictionary where keys are layer names and values are dictionaries with hooks and other information.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments, including 'model' which is the model to infer.

        Returns:
            dict:
                A dictionary containing the computed gradients, saliency map, and block gradient.

        Raises:
            AssertionError:
                If 'model' is not provided in kwargs.

        Example:
            >>> import torch
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> from explainer import Explainer, ModelExplanationTask
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> input_text = "Hello, my dog is cute"
            >>> input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
            >>> with Explainer(module=model, method="Gradient") as explain:
            >>>      model.eval()
            >>>      output = model.generate(**input_ids.clone())
            >>>      param = {
            >>>      "tokenizer": tokenizer,
            >>>      "output": output,
            >>>      "task_type": ModelExplanationTask.GENERATIVE_TEXT_CHAT,
            >>>      "input_ids": input_ids
            >>>      result = explain.compute(**param)
        """
        predict_result = LossFunctionCalculate()(**kwargs)
        predict_result.backward(retain_graph=True)

        for layer, modules in model.named_modules():
            output = layers[layer]["hook"].output
            self.gradients[layer] = {}
            if isinstance(output, List):
                self.gradients[layer]["output"] = output[0]
            elif isinstance(output, torch.Tensor):
                self.gradients[layer]["output"] = output
            else:
                self.gradients[layer]["output"] = None

            self.gradients[layer].update({"parent": layers[layer]["parent"]})
            self.gradients[layer].update({"level": layers[layer]["level"]})
            layers[layer]["hook"]

        saliency_map, gradient_block = Postprocessor(self.gradients)()
        self.result.update(
            {
                "gradient": self.gradients,
                "saliency_map": saliency_map,
                "block_gradient_x_input": gradient_block,
            }
        )
        return self.result
