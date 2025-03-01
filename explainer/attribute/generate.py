from typing import Optional
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
import torch
from explainer.attribute import LogitsHandler
from explainer.extra import get_logger
from explainer.attribute.attention_overwrite_module import overwrite_module

logger = get_logger(__name__)

# task_type correspond loss function
LOSS_FUN_MAP = {
    "question_answering": nn.CrossEntropyLoss(ignore_index=-100),
    "translation": nn.CrossEntropyLoss(ignore_index=-100),
    "summarizations": nn.CrossEntropyLoss(ignore_index=-100),
    "generative_text_chat": nn.CrossEntropyLoss(ignore_index=-100),
    "generative_text_chat_assistant": nn.CrossEntropyLoss(ignore_index=-100),
}


class ExplainerHandler:
    """
    A class to handle the generation of gradients for chatbot models.

    Args:
        model (PreTrainedModel):
            The main pretrained model to use.
        assistant_model (PreTrainedModel, optional):
            An optional assistant model to speed up inference.

    Returns:
        PreTrainedModel:
            The modified model.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from explainer import Explainer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> explainer_handler = ExplainerHandler(model)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        assistant_model: Optional[PreTrainedModel] = None,
    ):
        self.model = model
        self.assistant_model = assistant_model
        self._initialize_retrieve_attention()

    def _initialize_retrieve_attention(self):
        """
        Initialize the retrieve_attention attribute by attempting to overwrite the model's module functions.
        If an error occurs during this process, retrieve_attention is set to None.
        """
        try:
            _, self.retrieve_attention = overwrite_module(model=self.model)
            logger.info("Successfully initialized retrieve_attention.")
        except Exception as e:
            self.retrieve_attention = None
            logger.info(f"Error in overwrite_module: {e}")

    def adapt_hf_model_for_rai(self):
        """
        Modifies the Huggingface model's methods to fit RAI (record and Analyze Inference).

        Returns:
            PreTrainedModel: The modified model.
        """
        self.logits_handler = LogitsHandler(self.model)
        if not isinstance(self.assistant_model, type(None)):
            logger.info("Using assistant model to speed up inference")
            self.model._get_candidate_generator = (
                self.logits_handler._get_record_matches_assisted_generator
            )
            self.model._assisted_decoding = (
                self.logits_handler._record_n_matches_assisted_decoding
            )
        else:
            self.model._get_logits_processor = (
                self.logits_handler._clone_get_logits_processor
            )
            self.model._contrastive_search = (
                self.logits_handler._enable_grad_contrastive_search
            )

        self.model.generate = self.logits_handler.generate

        return self.model

    def generate_parameter(
        self, tokenizer, output, task_type, labels: Optional[torch.Tensor] = None, **kwargs
    ):
        """
        Prepares the parameters for gradient/attention generation based on the
        task_type and model outputs.

        Args:
            tokenizer:
                The tokenizer used for processing text.
            output:
                The model output.
            task_type (str):
                The task_type type.
            labels (torch.Tensor, optional):
                The labels tensor.
            **kwargs:
                Additional keyword arguments.

        Returns:
            dict:
                The parameters required for gradient generation.
        """
        param = {
            "tokenizer": tokenizer,
            "output": output,
            "labels": labels,
            "task_type": task_type,
        }
        param.update(kwargs)

        if not isinstance(self.assistant_model, type(None)):
            param.update(
                {
                    "n_matches": self.logits_handler.record_num_matches,
                    "task_type": "generative_text_chat_assistant",
                }
            )

        param.update(
            {
                "loss_function": LOSS_FUN_MAP[task_type],
                "attention_function": self.retrieve_attention,
            }
        )
        return param

    def __call__(self):
        if isinstance(self.model, PreTrainedModel):
            return self.adapt_hf_model_for_rai()
        else:
            return self.model
