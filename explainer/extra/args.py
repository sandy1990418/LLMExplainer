from dataclasses import dataclass, field
from typing import Optional, List, Union, Callable
from .constants import ModelExplanationTask
import torch.nn as nn


@dataclass
class TokenizerArgument:
    max_length: Optional[int] = field(
        default=150, metadata={"description": "Maximum length of the input sequence"}
    )
    tokenize: Optional[bool] = field(
        default=True,
        metadata={"description": "Whether to apply tokenization to the input"},
    )
    return_tensors: Optional[str] = field(
        default="pt",
        metadata={
            "description": "The format of the returned tensors, typically `pt` for PyTorch."
        },
    )
    add_generation_prompt: Optional[bool] = field(
        default=True,
        metadata={
            "description": "Whether to add a generation-specific prompt for the model"
        },
    )


@dataclass
class GenerationArgument:
    max_new_tokens: Optional[int] = field(
        default=320, metadata={"description": "LLM output maximum token number"}
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={
            "description": "If set to True, the model will sample from the top K most likely tokens."
        },
    )
    top_k: Optional[float] = field(default=1, metadata={"description": "top k"})
    top_p: Optional[float] = field(default=0.9, metadata={"description": "top p"})
    return_dict_in_generate: Optional[bool] = field(
        default=True,
        metadata={
            "description": "Whether to return a dictionary in the generate method."
        },
    )
    output_logits: Optional[bool] = field(
        default=True,
        metadata={
            "description": "Whether to output the logits in the generate method."
        },
    )
    num_beams: Optional[int] = field(
        default=1, metadata={"description": "Number of beams for beam search."}
    )
    use_cache: Optional[bool] = field(
        default=True,
        metadata={"description": "Whether to use cache in the generate method."},
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={
            "description": "Penalty to apply to repeated tokens in the generate method."
        },
    )


@dataclass
class ExplainationArgument:
    layer_names: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"],
        metadata={
            "description": "The name or naming pattern of attributes of self.module where queries are \
                located. Multiple names/patterns are supported."
        },
    )
    positions: Optional[List[int]] = field(
        default_factory=lambda: [0, 0, 0],
        metadata={
            "description": "The position of the query in the input/output arguments. Should be an integer.\
                If there's only one argument, position is 0."
        },
    )
    is_input: Optional[bool] = field(
        default=False,
        metadata={
            "description": "Specifies whether queries represent the input of the given layer.\
                If True, queries are considered input; if False, they represent outputs."
        },
    )
    strict: Optional[bool] = field(
        default=False,
        metadata={
            "description": "If True, the function will search for a layer with an exact name matching 'layer_name'. \
                If False, it will find layers with naming patterns matching 'layer_name'. Defaults to False."
        },
    )
    labels: Optional[str] = field(
        default=None,
        metadata={
            "description": "The ground truth label associated with the task, used for calculating explanations or losses."
        },
    )
    loss_function: Optional[Union[nn.Module, Callable]] = field(
        default=nn.CrossEntropyLoss,
        metadata={
            "description": "The loss function used during training, defining how the model's predictions are penalized for being incorrect. Commonly CrossEntropyLoss."
        },
    )
    task_type: Optional[Union[str, ModelExplanationTask]] = field(
        default=ModelExplanationTask.GENERATIVE_TEXT_CHAT,
        metadata={
            "description": "Type of task for the LLM. Possible types include: 'question_answering' (question answering), 'translation', 'summarizations', 'generative_text_chat', 'generative_text_chat_assistant'."
        },
    )
    input_shape: int = field(
        default=None,
        metadata={
            "description": "The shape of the input sequence to the model, typically specifying the sequence length or dimension."
        },
    )
