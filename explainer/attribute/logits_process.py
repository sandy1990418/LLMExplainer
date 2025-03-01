from typing import Callable, Dict, List, Optional, Any
import torch
from transformers.utils import add_start_docstrings
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.modeling_utils import PreTrainedModel

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
  return scores Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class CloningScoresLogitsProcessor(LogitsProcessor):
    """
    A logits processor that clones the prediction scores.

    This processor is useful for keeping a copy of the original scores before any modifications.
    """

    def __init__(self):
        super().__init__()

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Clone the prediction scores.

        Args:
            input_ids (torch.LongTensor): Indices of input sequence tokens in the vocabulary.
            scores (torch.FloatTensor): Prediction scores of a language modeling head.

        Returns:
            torch.FloatTensor: The cloned prediction scores.
        """
        scores_clone = scores.clone()
        return scores_clone


class RecordMatchesAssisted(AssistedCandidateGenerator):
    """
    A candidate generator that records the number of matches and new logits during assisted decoding.

    Args:
        input_ids (torch.LongTensor):
            Indices of input sequence tokens in the vocabulary.
        assistant_model (PreTrainedModel):
            The assistant model used for assisted decoding.
        generation_config (GenerationConfig):
            The generation configuration.
        logits_processor (LogitsProcessorList):
            The list of logits processors.
        model_kwargs (Dict):
            Additional model-specific arguments.
        inputs_tensor (torch.Tensor, optional):
            Additional input tensor.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        logits_processor: "LogitsProcessorList",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
    ):
        super(RecordMatchesAssisted, self).__init__(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            logits_processor=logits_processor,
            inputs_tensor=inputs_tensor,
        )
        self.record_num_matches = []
        self.record_new_logits = []

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):
        """
        Update the candidate strategy by recording the number of matches and new logits.

        Args:
            input_ids (torch.LongTensor):
                Indices of input sequence tokens in the vocabulary.
            scores (torch.FloatTensor):
                Prediction scores of a language modeling head.
            num_matches (int):
                Number of matches found.

        Returns:
            Any:
                The result of the parent class's update_candidate_strategy method.
        """
        self.record_num_matches += (num_matches,)
        self.record_new_logits += (input_ids,)
        return super().update_candidate_strategy(input_ids, scores, num_matches)


class LogitsHandler(GenerationMixin):
    """
    A generation mixin that adds cloning of logits processors and supports recording matches during assisted decoding.

    Args:
        model (PreTrainedModel):
            The model to apply the logits processor to.
    """

    def __init__(self, model):
        super().__init__()
        self._merge_criteria_processor_list = super()._merge_criteria_processor_list
        self._get_logits_processor_hf = super()._get_logits_processor
        self.generate_hf = super().generate
        self.model = model
        self._assisted_decoding = model._assisted_decoding
        self.record_num_matches = []
        self.record_new_logits = []

    def _clone_get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        """
        Get the logits processor list with the cloning processor added.

        Args:
            generation_config (GenerationConfig):
                The generation configuration.
            input_ids_seq_length (int):
                The length of the input sequence.
            encoder_input_ids (torch.LongTensor):
                The input tensor for the encoder.
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                The function that filters the vocabulary.
            logits_processor (Optional[LogitsProcessorList]):
                The list of logits processors.
            model_kwargs (Optional[Dict[str, Any]], optional):
                Additional model-specific arguments.
            negative_prompt_ids (Optional[torch.Tensor], optional):
                The tensor containing the negative prompts.
            negative_prompt_attention_mask (Optional[torch.Tensor], optional):
                The attention mask for the negative prompts.

        Returns:
            LogitsProcessorList:
                The list of logits processors.

        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> from attribute.logits_process import *
            >>> model = AutoModelForCausalLM.from_pretrained('gpt2)
            >>> model._get_logits_processor_hf = LogitsHandler(model)._clone_get_logits_processor

        Constraint:
            Algorithms based on beam_search method are not suitable.
        """
        processors = self._get_logits_processor_hf(
            generation_config,
            input_ids_seq_length,
            encoder_input_ids,
            prefix_allowed_tokens_fn,
            logits_processor,
            model_kwargs,
            negative_prompt_ids,
            negative_prompt_attention_mask,
        )

        processors.insert(0, CloningScoresLogitsProcessor())

        return processors

    def _enable_grad_contrastive_search(self, input_ids: torch.LongTensor, **kwargs):
        """
        Enable gradient contrastive search.

        Args:
            input_ids (torch.LongTensor):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            _type_:
                The result of the parent class's contrastive search method.

        Example:
            from transformers import AutoModelForCausalLM
            from attribute.logits_process import *
            model = AutoModelForCausalLM.from_pretrained('gpt2)
            model._contrastive_search = LogitsHandler(model)._clone_contrastive_search
        """
        return super()._contrastive_search.__wrapped__(self.model, input_ids, **kwargs)

    def _get_record_matches_assisted_generator(
        self,
        generation_config: GenerationConfig,
        input_ids: torch.LongTensor,
        inputs_tensor: torch.Tensor,
        assistant_model: "PreTrainedModel",
        logits_processor: LogitsProcessorList,
        model_kwargs: Dict,
    ):
        """
        Get the assisted candidate generator and record matches during assisted decoding.

        Args:
            generation_config (GenerationConfig):
                The generation configuration.
            input_ids (torch.LongTensor):
                Indices of input sequence tokens in the vocabulary.
            inputs_tensor (torch.Tensor):
                Input tensor for the generation process.
            assistant_model (PreTrainedModel):
                The assistant model used for assisted decoding.
            logits_processor (LogitsProcessorList):
                The list of logits processors.
            model_kwargs (Dict):
                Additional model-specific arguments.

        Returns:
            RecordMatchesAssisted:
                The assisted candidate generator with recording capabilities.
        """
        candidate_generator = RecordMatchesAssisted(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
        )

        return candidate_generator

    def _record_n_matches_assisted_decoding(self, *args, **kwargs):
        """
        Record the number of matches and new logits during assisted decoding.

        Args:
            *args: Positional arguments used for the `assisted_decoding` method from Huggingface.
            **kwargs: Keyword arguments used for the `assisted_decoding` method from Huggingface.

        Returns:
            Any: The result of the parent class's assisted decoding method.
        """
        result = self._assisted_decoding(*args, **kwargs)
        self.record_num_matches = kwargs["candidate_generator"].record_num_matches
        self.record_new_logits = kwargs["candidate_generator"].record_new_logits

        return result

    def generate(self, **kwargs):
        """
        Generate sequences of token ids.

        Args:
            **kwargs:
                Additional generation-specific arguments.

        Returns:
            torch.LongTensor:
                The generated sequences of token IDs.
        """
        return self.generate_hf.__wrapped__(self.model, **kwargs)
