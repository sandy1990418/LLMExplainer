from explainer.extra.logging import get_logger
from typing import Optional, Any, Tuple, Dict, Union
import torch
from explainer.loader import Loader
import re

logger = get_logger(__name__)


# The maximum length of a feature, following Hugging Face's settings.
DEFAULT_MAX_FEATURE_LENGTH = 384
# The authorized overlap between two part of the context when splitting
# it is needed, following Hugging Face's settings.
DEFAULT_DOC_STRIDE_LENGTH = 128


class Preprocessor(Loader):
    """
    Preprocessor class is used to preprocess the input data for the model.
    """
    def __init__(
        self,
        preprocess_input: Optional[Any] = None,
        input_text: Optional[Union[str, dict]] = None,
        *args,
        **kwargs,
    ):
        """Initialize the Preprocessor class.

        Args:
            preprocess_input (Optional[Any], optional):
                Preprocessed input data. Defaults to None.
            input_text (Optional[Union[str, dict]], optional):
                Input text or dictionary. Defaults to None.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        # check the preprocess
        self.preprocess_input = preprocess_input
        self.inputs = input_text
        self.config = super().config()

    def preprocess_function(self, *args, **kwargs):
        """
        Preprocesses the input data for the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[Dict[str, Any], tuple]: The preprocessed input data.

        Example:
            >>> from explainer.attribute import Preprocessor
            >>> input_text = "Hello, my dog is cute"
            >>> task_type = "translation
            >>> model_name_or_path = "t5-base"
            >>> model_config_path = "example/yaml/translation.yaml"
            >>> preprocess_result = Preprocessor(input_text=input_text,
            >>>                                  task_type=task_type,
            >>>                                  tokenizer=model_name_or_path,
            >>>                                  model_name_or_path=model_name_or_path,
            >>>                                  model_config_path=model_config_path)
            >>> encode = preprocess_result.preprocess_function()
        """
        if isinstance(self.preprocess_input, type(None)):
            funtion_map = {
                "question_answering": self.qa_task_preprocess,
                "translation": self.translation_summarizations_task_preprocess,
                "summarizations": self.translation_summarizations_task_preprocess,
            }
            preprocess_method = funtion_map[self.task_type]
            token_type = (
                str(self.tokenizer.tokenize).split("of ")[1].split("(")[0]
                if isinstance(self.config, type(None))
                else self.config["tokenizer_config"]["tokenizer_class"]
            )
            inputs = preprocess_method(
                token_type=token_type, input_ids=self.inputs, *args, **kwargs
            )
        else:
            inputs = self.preprocess_input
        return inputs

    @staticmethod
    def get_start_and_end_position(
        input_convert2token, input_text
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Get the start and end position of the answer.

        Args:
            input_convert2token (Any):
                The input data converted to tokens.
            input_text (Any):
                The input text data.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]:
                The start and end positions of the answer.

        Ref : https://huggingface.co/docs/transformers/tasks/question_answering
        """
        # Get Start and End position
        start_positions = []
        end_positions = []
        for i, offset in enumerate(input_convert2token["offset_mapping"]):
            start_char = input_text["answers"]["answer_start"][0]
            end_char = input_text["answers"]["answer_start"][0] + len(
                input_text["answers"]["text"][0]
            )
            sequence_ids = input_convert2token.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        # get a Mask for masking [PAD]
        mask = [i != 1 for i in input_convert2token.sequence_ids()]
        # Unmask the [CLS] token
        mask[0] = False
        mask = torch.tensor(mask)[None]
        return (
            torch.LongTensor(start_positions),
            torch.LongTensor(end_positions),
            mask,
        )

    def qa_task_preprocess(
        self,
        token_type: str = None,
        input_ids: dict = None,
        doc_stride: Optional[int] = None,
        model_max_length: Optional[int] = None,
        padding: Optional[Any] = "max_length",
        return_token_type_ids: Optional[bool] = True,
        return_overflowing_tokens: Optional[bool] = True,
        return_offsets_mapping: Optional[bool] = True,
        bert_add_special_tokens: Optional[dict] = {"pad_token": "[PAD]"},
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], tuple]:
        """ Preprocess the input data for the question-answering task.

        Args:

            token_type (str, optional):
                The type of tokenizer. Defaults to None.
            input_ids (dict, optional):
                The input IDs for the model. Defaults to None.
            doc_stride (int, optional):
                The authorized overlap between two part of the context.
            model_max_length (int, optional):
                The maximum length of a feature (question and context). Defaults to None.
            padding (Any, optional):
                Padding strategy. Defaults to "max_length".
            return_token_type_ids (bool, optional):
                Whether to return token type IDs. Defaults to True.
            return_overflowing_tokens (bool, optional):
                Whether to return overflowing tokens. Defaults to True.
            return_offsets_mapping (bool, optional):
                Whether to return offsets mapping. Defaults to True.
            bert_add_special_tokens (dict, optional):
                Special tokens for BERT. Defaults to {"pad_token": "[PAD]"}.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            Union[Dict[str, Any], tuple]:
                The preprocessed input data.

        """
        # Dealing with long docs:
        # The maximum length of a feature (question and context)
        if model_max_length is None:
            model_max_length = min(
                self.tokenizer.model_max_length, DEFAULT_MAX_FEATURE_LENGTH
            )

        # The authorized overlap between two part of the context when splitting it is needed.
        if doc_stride is None:
            doc_stride = min(model_max_length // 2, DEFAULT_DOC_STRIDE_LENGTH)

        question_first = self.tokenizer.padding_side == "right"

        if "gpt" in token_type.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token

        elif (
            "bert" in token_type.lower()
        ):  # or 'Bert' in model.config.self.tokenizer_class
            self.tokenizer.add_special_tokens(bert_add_special_tokens)
        inputs = self.tokenizer(
            text=input_ids["question"] if question_first else input_ids["context"],
            text_pair=input_ids["context"]
            if input_ids["context"]
            else input_ids["question"],
            return_tensors="pt",
            truncation="only_second" if question_first else "only_first",
            padding=padding,
            max_length=model_max_length,
            stride=doc_stride,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_offsets_mapping=return_offsets_mapping,
            *args,
            **kwargs,
        )
        if "answers" in input_ids.keys():
            (
                inputs["start_positions"],
                inputs["end_positions"],
                inputs["mask"],
            ) = self.get_start_and_end_position(inputs, input_ids)

        return inputs

    def translation_summarizations_task_preprocess(
        self,
        token_type: str = None,
        input_ids: dict = None,
        truncation: Optional[bool] = True,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], tuple]:
        """Preprocess the input data for the translation or summarizations task.

        Args:
            token_type (str, optional):
                The type of tokenizer. Defaults to None.
            input_ids (dict, optional):
                The input IDs for the model. Defaults to None.
            truncation (bool, optional):
                Whether to truncate the input. Defaults to True.
            return_token_type_ids (bool, optional):
                Whether to return token type IDs. Defaults to True.
            return_attention_mask (bool, optional):
                Whether to return attention mask. Defaults to True.
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            Union[Dict[str, Any], tuple]:
                The preprocessed input data.
        """
        inputs = [
            self.config["prefix"] + input_text[self.config["source"]]
            for input_text in input_ids
        ]
        targets = (
            [target_text[self.config["target"]] for target_text in input_ids]
            if not isinstance(self.config["target"], type(None))
            else None
        )
        if bool(re.search(r"t5|bert|marian", token_type.lower())):
            inputs = self.tokenizer(
                text=inputs,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
                if "model_max_length" not in self.config.keys()
                else self.config["model_max_length"],
                padding=True,
                truncation=truncation,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                *args,
                **kwargs,
            )
            # Ref : https://github.com/huggingface/transformers/issues/18455
            assert not isinstance(
                targets, type(None)
            ), "Please check target is not None"
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(
                    text=targets, return_tensors="pt", *args, **kwargs
                )
            inputs["labels"] = targets["input_ids"]
            logger.info("Get Translation or summarizations task")

            return inputs

    def decode_text(self, inputs):
        """Decode the input text.

        Args:
            inputs (dict):
                The token inputs.

        Returns:
            list:
                The decoded text.
        """
        decode_text = []
        for i in range(inputs["input_ids"][0].shape[0]):
            decode_text.extend([self.tokenizer.decode(inputs["input_ids"][0][i])])

        return decode_text

    def __call__(
        self,
        keep_list: Optional[list] = ["input_ids", "token_type_ids", "attention_mask"],
        *args,
        **kwargs,
    ):
        """Preprocesses the input data for the model.

        Args:
            keep_list (Optional[list], optional):
                The list of keys to keep. Defaults to ["input_ids", "token_type_ids", "attention_mask"].
            *args:
                Variable length argument list.
            **kwargs:
                Arbitrary keyword arguments.

        Returns:
            tuple:
                The preprocessed input data.

        Example:
            >>> from explainer.attribute import Preprocessor
            >>> input_text = "Hello, my dog is cute"
            >>> task_type = "translation
            >>> model_name_or_path = "t5-base"
            >>> model_config_path = "example/yaml/translation.yaml"
            >>> preprocess_result = Preprocessor(input_text=input_text,
            >>>                                  task_type=task_type,
            >>>                                  tokenizer=model_name_or_path,
            >>>                                  model_name_or_path=model_name_or_path,
            >>>                                  model_config_path=model_config_path)
            >>> inputs_all, inputs = preprocess_result(keep_list = ['input_ids'])
        """
        inputs_all = self.preprocess_function(*args, **kwargs)
        inputs_model = {}
        try:
            for key in keep_list:
                inputs_model[key] = inputs_all[key]
            logger.info(f"Selected input keys: {', '.join(keep_list)}")
        except Exception:
            logger.warning("All input keys are selected.")

        return inputs_all, inputs_model


class ContextManager(Preprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_result = Preprocessor(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        print("--enter--")
        print(self.test_result.config())
        print(self.test_result.model)

    def __exit__(self, *args, **kwargs):
        print("--exit--")
        self.test_result.model = []
        print(self.test_result.model)
