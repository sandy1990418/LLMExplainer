from typing import Optional, Union, Callable
import torch.nn as nn
import torch
from transformers.utils import ModelOutput

TASK_LIST = {"question_answering", "translation", "summarizations", "generative_text_chat", "generative_text_chat_assistant"}


class OutputPostProcessing:
    """
    A class to handle the post-processing of outputs based on the specified task.

    Args:
        task (str):
            The task type, must be one of the predefined tasks in TASK_LIST.

    Raises:
        AssertionError:
            If the task is not one of the tasks in TASK_LIST.

    Returns:
        dict:
            The processed output.
    """
    def __init__(self, task_type):
        assert task_type in TASK_LIST  # [*TASK_LIST.keys()]
        self.task_type = task_type

    def __call__(self, *args, **kwargs) -> dict:
        if self.task_type == "question_answering":
            post_process = QuestionAnsweringPostProcessing()
        elif self.task_type != "question_answering":
            post_process = TextBasicPostProcessing()

        return post_process(*args, **kwargs)


class QuestionAnsweringPostProcessing:
    """
    A class to handle the post-processing for question answering tasks.

    Args:
        output (ModelOutput, optional):
            The model output containing start and end logits.
        start_logit (torch.Tensor, optional):
            The start logits tensor.
        end_logit (torch.Tensor, optional):
            The end logits tensor.
        start_positions (torch.Tensor, optional):
            The start positions tensor.
        end_positions (torch.Tensor, optional):

    Raises:
        AssertionError:
            If `output` does not contain `start_logits` or `end_logits`.
        AssertionError:
            If `start_logit` or `end_logit` is not two-dimensional.
        AssertionError:
            If `start_positions` or `end_positions` is not provided.

    Returns:
        dict:
            The processed start and end logits, and start and end positions.
    """
    def process_hf_output(self, output):
        """
        Processes the output when provided as a ModelOutput.

        Args:
            output (ModelOutput): The model output containing start and end logits.

        Returns:
            tuple: The start and end logits.

        Raises:
            AssertionError: If `output` does not contain `start_logits` or `end_logits`.
        """
        assert "start_logits" in [
            *output.keys()
        ], "Please provide Start logits for calculating loss function"
        assert "end_logits" in [
            *output.keys()
        ], "Please provide End logits for calculating loss function"
        return output.start_logits, output.end_logits

    def process_non_hf_output(self, start_logit, end_logit):
        """
        Processes the logits when not provided as a ModelOutput.

        Args:
            start_logit (torch.Tensor): The start logits tensor.
            end_logit (torch.Tensor): The end logits tensor.

        Returns:
            tuple: The start and end logits.

        Raises:
            AssertionError: If `start_logit` or `end_logit` is not two-dimensional.
        """
        assert (
            len(start_logit.shape) == 2
        ), f"Dimensio of start_logits is {start_logit.shape}, "
        assert (
            len(end_logit.shape) == 2
        ), f"Dimensio of start_logits is {end_logit.shape}, "
        return start_logit, end_logit

    def process_labels(self, start_positions, end_positions):
        """
        Processes the start and end positions labels.

        Args:
            start_positions (torch.Tensor): The start positions tensor.
            end_positions (torch.Tensor): The end positions tensor.

        Returns:
            tuple: The processed start and end positions.
        """
        assert not isinstance(
            start_positions, type(None)
        ), "Please provide start_positions."
        assert not isinstance(
            end_positions, type(None)
        ), "Please provide end_positions."

        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        return start_positions, end_positions

    def __call__(
        self,
        output: Optional[ModelOutput] = None,
        start_logit: Optional[torch.Tensor] = None,
        end_logit: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """
        Processes the inputs based on the provided arguments.

        Args:
            output (ModelOutput, optional): The model output.
            start_logit (torch.Tensor, optional): The start logits tensor.
            end_logit (torch.Tensor, optional): The end logits tensor.
            start_positions (torch.Tensor, optional): The start positions tensor.
            end_positions (torch.Tensor, optional): The end positions tensor.

        Returns:
            dict: The processed start and end logits, and start and end positions.
        """
        if isinstance(output, ModelOutput):
            start_logits, end_logits = self.process_hf_output(output)
        else:
            assert start_logit is not None, "Please provide start_logits"
            assert end_logit is not None, "Please provide end_logits"
            start_logits, end_logits = self.process_non_hf_output(
                start_logit, end_logit
            )

        start_positions, end_positions = self.process_labels(
            start_positions, end_positions
        )

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }


class TextBasicPostProcessing:
    """
    A class to handle summarzation/translation or chat text post-processing tasks.
    """
    def check_hf_logits_type(self, output):
        if 'generation' in str(type(output)):
            output.logits = torch.stack(output.logits, dim=0)
        else:
            pass
        return output

    def process_hf_logits(self, output, **kwargs):
        """
        Checks the type of Huggingface logits and processes accordingly.

        Args:
            output: The output to check.

        Returns:
            The processed output.

        Raises:
            AssertionError: If `output` does not contain `logits`.
            AssertionError: If `n_matches` is required but not provided in `kwargs`.
        """
        assert hasattr(
            output, "logits"
        ), "The Output don't contain logits, please add `return_logits` in generate output"

        if "n_matches" in kwargs:
            assert not isinstance(
                kwargs["n_matches"], type(None)
            ), "Please add `n_matches` for calculating loss of Assisted Generation."

            output.logits = [
                logits[0][: kwargs["n_matches"][idx].item() + 1, :]
                for idx, logits in enumerate(output.logits)
            ]
            output.logits = torch.cat(output.logits, axis=0)
        else:
            output = self.check_hf_logits_type(output)

        # reshape output.logits to (input dim, vo)
        logits = output.logits.view(-1, output.logits.size(-1))
        return logits

    def process_torch_logits(self, output):
        """
        Processes PyTorch logits.

        Args:
            output (torch.Tensor): The logits tensor.

        Raises:
            AssertionError: If `output` is not a torch.Tensor.
            AssertionError: If `output` is not two-dimensional.
        """
        assert isinstance(
            output, torch.Tensor
        ), "The boolbox currently support only Huggingface's output and Pytorch."
        assert (
            len(output.shape) == 2
        ), f"The dimension of output is {output.shape}, it should be two-dimensional (Sequence_length, Vocab_size)."
        return output

    def process_dict_logits(self, output, logits_name):
        """
        Processes logits from a dictionary.

        Args:
            output (dict): The dictionary containing logits.
            logits_name (str): The key for accessing logits in the dictionary.

        Returns:
            torch.Tensor: The processed logits tensor.

        Raises:
            AssertionError: If `output` does not contain `logits_name` or if the logits are `None`.
        """
        # check output contain logits and is not None
        assert output.has_key(logits_name) and not isinstance(
            output[logits_name], type(None)
        ), "The Output don't contain logits, please check your result is not None or provide correct logits_name."
        return self.process_torch_logits(output[logits_name])

    def _one_hot_batched(self, calculate_logits_shape, vocab_size, device):
        return torch.zeros(calculate_logits_shape, vocab_size, device=device)

    def _make_length_equal(self, logits, labels):
        """
        Ensures the lengths of logits and labels are equal.

        Args:
            logits (torch.Tensor): The logits tensor.
            labels (torch.Tensor): The labels tensor.

        Returns:
            dict: A dictionary containing padded logits and labels.
        """
        assert isinstance(labels, torch.Tensor), "labels should be Pytorch."
        assert (
            len(labels.shape) == 1
        ), f"The dimension of ground truth is {logits.shape}, it should be one-dimensional (Sequence_length)."

        padding_result = {}
        # considering cases where the inference result differs from the ground truth
        if labels.shape[-1] >= logits.shape[0]:
            one_hot_batch = self._one_hot_batched(
                calculate_logits_shape=labels.shape[0],
                vocab_size=logits.shape[-1],
                device=logits.device,
            )
            one_hot_batch[: logits.shape[0], :] = logits
            padding_result.update(
                {"logits": one_hot_batch, "labels": labels.to(logits.device)}
            )
        else:
            one_hot_batch = torch.full(
                (1, logits.shape[0]),
                -100,
                device=logits.device,
            )
            one_hot_batch[:, : labels.shape[-1]] = labels
            padding_result.update({"logits": logits, "labels": one_hot_batch.view(-1)})
        return padding_result

    def __call__(
        self,
        output: Optional[Union[ModelOutput, dict, torch.tensor]] = None,
        labels: Optional[torch.tensor] = None,
        logits_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Processes the inputs based on the provided arguments.

        Args:
            output (Union[ModelOutput, dict, torch.Tensor], optional):
                The model output or logits.
            labels (torch.Tensor, optional):
                The labels tensor.
            logits_key (str, optional):
                The key for accessing logits in a dictionary.

        Returns:
            dict:
                The processed logits and labels.

        Raises:
            AssertionError: If `output` is None.
        """
        assert not isinstance(
            output, type(None)
        ), "Output cannot be None. Please provide the output"

        if isinstance(output, ModelOutput):
            logits = self.process_hf_logits(output, **kwargs)
        elif isinstance(output, dict):
            assert not isinstance(
                logits_key, type(None)
            ), "When the output is a dict, logits_key cannot be None. Please provide the logits_key."
            logits = self.process_dict_logits(output, logits_key)
        elif isinstance(output, torch.Tensor):
            logits = self.process_torch_logits(output)
        else:
            raise ValueError("Please ...")

        padding_result = self._make_length_equal(logits, labels)
        return padding_result


class LossFunctionCalculate:
    """
    Calculates the loss based on different tasks. The supported tasks include:
    - "question_answering" (Question Answering)
    - "translation" (Language Translation)
    - "summarizations" (Text summarizations)
    - "generative_text_chat" (Conversational Agents)
    - "assisted_decoding" (Assisted Decoding using an assistant model)

    Args:
        loss_function (Union[nn.Module, Callable], optional):
            The loss function to use.
        task (str):
            The task type.

    Raises:
        AssertionError:
            If the task is not one of the tasks in TASK_LIST.

    Returns:
        torch.Tensor:
            The calculated loss.
    """
    def _calculate(
        self,
        loss_function: Optional[Union[nn.Module, Callable]] = nn.CrossEntropyLoss(
            ignore_index=-100
        ),
        task_type: str = None,
        *args,
        **kwargs,
    ):
        postprocess = OutputPostProcessing(task_type=task_type)(*args, **kwargs)

        if task_type == "question_answering":
            # we refer to Huggingface's method.
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = postprocess["start_logits"].size(1)
            start_positions = postprocess["start_positions"].clamp(0, ignored_index)
            end_positions = postprocess["end_positions"].clamp(0, ignored_index)
            start_loss = loss_function(postprocess["start_logits"], start_positions)
            end_loss = loss_function(postprocess["end_logits"], end_positions)
            ce_loss = (start_loss + end_loss) / 2
        else:
            ce_loss = loss_function(postprocess["logits"], postprocess["labels"])

        return ce_loss

    def __call__(self, *args, **kwargs):
        return self._calculate(*args, **kwargs)
