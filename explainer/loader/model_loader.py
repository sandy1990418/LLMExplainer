import os
import yaml
from explainer.extra.logging import get_logger
from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)
from typing import Optional, Union
import torch.nn as nn
import json
import shutil

# TODO: AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoModelForMaskedLM

logger = get_logger(__name__)


def load_config(model_config):
    path = os.path.dirname(os.path.abspath("tmp"))
    configs = yaml.safe_load(
        open(os.path.join(path, f"{model_config}"), encoding="utf8")
    )
    return configs


def model_from_pretrained(
    task_type: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    load_torch: Optional[dict] = None,
):
    autoclass = {
        "question_answering": AutoModelForQuestionAnswering,
        "translation": AutoModelForSeq2SeqLM,
        "summarizations": AutoModelForSeq2SeqLM,
        "causal": AutoModelForCausalLM,
    }
    if isinstance(load_torch, type(None)):
        if task_type in autoclass.keys():
            model_cls = autoclass[task_type]
        else:
            model_cls = AutoModel
        model = model_cls.from_pretrained(model_name_or_path)
    else:
        model = load_torch["model"]
    return model


def load_token_config(tokenizer):
    path = os.path.join(os.path.dirname(os.path.abspath("tmp")), "loader/token_temp")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    tokenizer.save_pretrained(path)
    with open(f"{path}/tokenizer_config.json", encoding="utf-8") as reader:
        token = json.load(reader)

    shutil.rmtree(path, ignore_errors=True)

    return token


class Loader:
    """A class to load the model and tokenizer for the explainer.

    Args:
        task_type (str):
            The task for which the model is being used.
        tokenizer (Any):
            The tokenizer or tokenizer name from Huggingface to associated with the LLM.
        model_name_or_path (nn.Module, str):
            The Language Model or  or model name from Huggingface to be explained.
        model_config_path (str):
            The path to the model configuration file.
        load_torch (dict):
            A dictionary containing the model and tokenizer to be loaded.

    Raises:
        ValueError:
            If the model or tokenizer is not specified.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from explainer import Loader
        >>> model = "gpt2"
        >>> tokenizer = "gpt2"
        >>> explainer_handler = Loader(model, tokenizer)

    """
    def __init__(
        self,
        task_type: Optional[str] = None,
        tokenizer: Union[PreTrainedTokenizer, str] = None,
        model_name_or_path: Union[nn.Module, str] = None,
        model_config_path: Optional[str] = None,
        load_torch: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """Initializes the Loader with the given model, tokenizer, and other parameters."""
        logger.info("Load model")
        if isinstance(model_name_or_path, nn.Module):
            self.model = model_name_or_path
        elif isinstance(model_name_or_path, str) or not isinstance(
            load_torch["model"], type(None)
        ):
            self.model = model_from_pretrained(
                task_type, model_name_or_path, load_torch
            )

        logger.info("Load Tokenizer")

        if not isinstance(tokenizer, str) and not isinstance(tokenizer, type(None)):
            self.tokenizer = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif not isinstance(load_torch["tokenizer"], type(None)):
            self.tokenizer = load_torch["tokenizer"]

        if not self.tokenizer or not self.model:
            raise ValueError(
                "You did not specify the model/tokenizer location, "
                "either from your own path or from Hugging Face; "
                "therefore, you cannot load the tokenizer and model correctly."
            )

        self.model_config_path = model_config_path
        logger.info(f"{self.model_config_path}")

        self.task_type = task_type
        logger.info(f"{self.task_type}")

    def config(self):
        """Returns the configuration of the model.

        Returns:
            dict:
                The configuration of the model.

        Example:
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> from explainer import Loader
            >>> model = "gpt2"
            >>> tokenizer = "gpt2"
            >>> explainer_handler = Loader(model, tokenizer)
            >>> explainer_handler.config()
        """
        if isinstance(self.model_config_path, type(None)):
            logger.warning(
                "The path of config is be assigned, the config will return None."
            )
            return None

        config = load_config(self.model_config_path)
        token_config = load_token_config(self.tokenizer)
        config["tokenizer_config"].update(token_config)
        return config
