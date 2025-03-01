import warnings
import torch
from torch import nn
from typing import List, Tuple, Union, Any, Optional
from explainer.attribute.generate import ExplainerHandler


class LayerHook:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooked = False

    def __enter__(self):
        self.explainer_handler = ExplainerHandler(self.model)
        model = self.explainer_handler()
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError("Layer already hooked.")

        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError("Layer is not hooked.")

        self.hooked = False
        self._unhook_impl()

        return self

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        raise NotImplementedError

    def _raw_to_tensor(
        self, raw: Any
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Retrieved torch.Tensor from raw data in tuple or list.

        Args:
            raw (Any): Any kinds of data.
        Return:
            tensor (torch.Tensor, List, Tuple)
        """
        # Multiple tensors
        if isinstance(raw, (Tuple, List)):
            tensor = []
            for x in raw:
                if isinstance(x, torch.Tensor):
                    if x.requires_grad is True:
                        x.retain_grad()
                    tensor.append(x)
            if len(tensor) == 1:
                tensor = tensor[0]
            elif len(tensor) == 0:
                tensor = None

        # Single tensor
        elif isinstance(raw, torch.Tensor):
            if raw.requires_grad is True:
                raw.retain_grad()
            tensor = raw

        # No data
        elif raw is None:
            tensor = None

        # Unrecognized data type
        else:
            message = f"Unsupported type {type(raw)}."
            warnings.warn(message, UserWarning)
            tensor = None
        return tensor


class ExplainerHook(LayerHook):
    def __init__(self, model: nn.Module, name: str, flow: List):
        super().__init__(model)
        self.tensors = {}
        self._input: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self._output: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self._num_input: int = 0
        self._num_output: int = 0
        self._counter: int = 0
        self._flow = flow
        self._name = name

    def _update_input(self, inputs):
        data = self._raw_to_tensor(inputs)
        if data is None:
            return

        if self._input is None:
            self._input = data
            self._update_num_input(data)
            return

        # no input to forward function
        if self._num_input == 0:
            pass

        # single input to forward function
        elif self._num_input == 1:
            if not isinstance(self._input, List):
                self._input = [self._input]
            self._input.append(data)

        # multiple inputs to forward function
        else:
            if not isinstance(self._input[0], List):
                self._input = [self._input]
            self._input.append(data)

    def _update_output(self, outputs):
        data = self._raw_to_tensor(outputs)
        if data is None:
            return

        if self._output is None:
            self._output = data
            self._update_num_output(data)
            return

        # forward function return no data
        if self._num_output == 0:
            pass

        # forward function return single data
        elif self._num_output == 1:
            if not isinstance(self._output, List):
                self._output = [self._output]
            self._output.append(data)

        # forward function return multiple data
        else:
            if not isinstance(self._output[0], List):
                self._output = [self._output]
            self._output.append(data)

    def _update_num_input(self, data):
        if isinstance(data, List):
            self._num_input = len(data)
        elif isinstance(data, torch.Tensor):
            self._num_input = 1
        else:
            self._num_input = 0

    def _update_num_output(self, data):
        if isinstance(data, List):
            self._num_output = len(data)
        elif isinstance(data, torch.Tensor):
            self._num_output = 1
        else:
            self._num_output = 0

    def _update_counter(self):
        self._counter += 1

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def num_input(self):
        return self._num_input

    @property
    def num_output(self):
        return self._num_output

    @property
    def counter(self):
        return self._counter

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def _find_target_dim(self, a, b):
        assert len(a.shape) == len(
            b.shape
        ), f"Unmatch tensor size: {a.shape} and {b.shape}."
        target_dim = None
        for dim in range(1, len(a.shape)):
            if a.shape[dim] != b.shape[dim]:
                target_dim = dim
                break
        assert target_dim, "Target dimension not found."
        return target_dim

    def _update_forward(self, model, inputs, outputs):
        self._flow.append(self._name)
        self._update_input(inputs)
        self._update_output(outputs)

    def _hook_impl(self):
        self._handle1 = self.model.register_forward_hook(self._update_forward)

    def _unhook_impl(self):
        self.tensors.clear()
        self._handle1.remove()
