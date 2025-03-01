import torch
import numpy as np
import pandas as pd
import json
from typing import Optional
import itertools
from explainer.extra.logging import get_logger
from transformers.utils import ModelOutput


logger = get_logger(__name__)

EMBEDDING_NAME = {
    "bert": ["word_embeddings"],
    "gpt": ["wte"],
    "Marian": ["shared"],
    "T5": ["embed_tokens"],
}


class Postprocessor(object):
    """
    A class for postprocessing gradient/attention information from a model.

    Args:
        gradient (Optional[dict]):
            The gradient information.
        embedding (Optional[str]):
            The embedding layer name. Defaults to None.
        *args:
            Variable length argument list.
        **kwargs:
            Arbitrary keyword arguments.

    Attributes:
        gradient (dict):
            The gradient information.
        embedding_name (list):
            A list of embedding layer names.
        embedding_layer_name (str):
            The name of the embedding layer used.
        input_embedding (torch.Tensor):
            The input embedding tensor.
        input_embedding_level (int):
            The level of the input embedding.

    Methods:
        convert_list_input2tensor(inputs):
            Converts a list input to a tensor if it is a list.

        get_embedding():
            Retrieves the embedding layer, its output, and level from the gradient information.

        saliency_map(abs=True):
            Generates a saliency map from the gradient information.

        gradient_x_input():
            Computes the gradient x input norm for intermediate layers.

        mean_attention_layers(qkv):
            Computes the mean attention values across layers.
    """

    def __init__(
        self, gradient: Optional[dict], embedding: Optional[str] = None, *args, **kwargs
    ):
        self.gradient = gradient
        self.embedding_name = list(itertools.chain(*list(EMBEDDING_NAME.values())))
        self.embedding_name.extend(
            embedding
        ) if embedding is not None else self.embedding_name

        (
            self.embedding_layer_name,
            self.input_embedding,
            self.input_embedding_level,
        ) = self.get_embedding()

    @staticmethod
    def convert_list_input2tensor(inputs):
        """
        Converts a list input to a tensor if it is a list.

        Args:
            inputs (list or torch.Tensor): The input data.

        Returns:
            torch.Tensor: The converted tensor.
        """
        if isinstance(inputs, list):
            return inputs[0]
        else:
            return inputs

    def get_embedding(self):
        """
        Retrieves the embedding layer, its output, and level from the gradient information.

        Returns:
            tuple:
                The embedding layer name, output tensor, and level.

        Raises:
            ValueError:
                If the embedding layer is not found in the gradient information.
        """
        input_embedding = None
        for layer_name in [*self.gradient]:
            if any(
                embedd_sub_name in layer_name for embedd_sub_name in self.embedding_name
            ):
                embedding_layer_name = layer_name
                input_embedding = self.gradient[layer_name]["output"]  # .grad
                input_embedding_level = self.gradient[layer_name]["level"]
                input_embedding_level = (
                    1
                    if isinstance(input_embedding_level, type(None))
                    else input_embedding_level
                )
                logger.info(
                    f"Success get model embedding layer, the layer's level is {input_embedding_level}"
                )
                break
        if isinstance(input_embedding, type(None)):
            raise ValueError(
                "Unable to locate the model's embedding layer. \
                 Please specify the model embedding layer name in the `embedding`."
            )
        return embedding_layer_name, input_embedding, input_embedding_level

    def saliency_map(self, abs: Optional[bool] = True):
        """
        Generates a saliency map from the gradient information.

        Args:
            abs (Optional[bool]):
                Whether to take the absolute value of the gradients. Defaults to True.

        Returns:
            list:
                The saliency map values.
        """
        target = self.gradient[self.embedding_layer_name]["output"]
        logger.debug(self.embedding_layer_name, target.shape, target.grad)
        input_grad = self.gradient[self.embedding_layer_name]["output"].grad

        if abs:
            embedding_saliency, _ = torch.max(
                torch.abs(self.convert_list_input2tensor(input_grad)),
                dim=-1,
            )
        else:
            embedding_saliency, _ = self.convert_list_input2tensor(input_grad)

        min_value = embedding_saliency.min(1, keepdim=True)[0]
        max_value = embedding_saliency.max(1, keepdim=True)[0]

        embedding_saliency -= min_value
        embedding_saliency /= max_value - min_value
        embedding_saliency = np.round(
            embedding_saliency[0].cpu().detach().tolist(), 3
        ).tolist()
        return embedding_saliency

    def gradient_x_input(self):
        """
        Computes the gradient x input norm for intermediate layers.

        Returns:
            list:
                The gradient x input norm values.
        """
        gradient_x_input_block = []
        for layer_name in [*self.gradient]:
            if (
                self.gradient[layer_name]["level"] == (self.input_embedding_level + 1)
                and "output" in self.gradient[layer_name].keys()
                and self.gradient[layer_name]["output"] is not None
                and "embed" not in layer_name
            ):
                self.gradient[layer_name]["output"] = self.convert_list_input2tensor(
                    self.gradient[layer_name]["output"]
                )
                logger.debug(f"{layer_name}, {self.gradient[layer_name]['level']}")
                intermediate_grad = self.gradient[layer_name]["output"].grad
                intermediate_grad = self.convert_list_input2tensor(intermediate_grad)

                grad_x_input_norm = torch.norm(
                    intermediate_grad * self.input_embedding, dim=-1
                )
                grad_x_input_norm -= grad_x_input_norm.min(1, keepdim=True)[0]
                grad_x_input_norm /= grad_x_input_norm.max(1, keepdim=True)[0]
                grad_x_input_norm = np.round(
                    grad_x_input_norm[0].cpu().detach().tolist(), 3
                ).tolist()
                # gradient_x_input_block.update({layer_name: grad_x_input_norm})
                gradient_x_input_block.append({"gradients": grad_x_input_norm[0]})
            elif (
                self.gradient[layer_name]["level"] == (self.input_embedding_level + 1)
                and "output" not in self.gradient[layer_name].keys()
            ):
                self.input_embedding_level += 1
            elif (
                self.gradient[layer_name]["level"] == (self.input_embedding_level + 1)
                and self.gradient[layer_name]["output"] is None
            ):
                self.input_embedding_level += 1
        return gradient_x_input_block

    def __call__(self):
        saliency_map = self.saliency_map()
        grad_x_input_norm_scale = self.gradient_x_input()

        return (saliency_map, grad_x_input_norm_scale)


class AttentionPostprocessor(object):
    """
    A class for postprocessing attention information from a model.

    Args:
        attention_dict (dict):
            Dictionary containing attention values.

    Methods:
        median_attention_layers(qkv):
            Computes the mean attention values across layers.

        _deal_block_matrix(matrix):
            Processes the attention matrix.

        _retrive_median_attention_dict(attention_dict):
            Retrieves the median attention values from the attention dictionary.

        _minimax_normalization(value):
            Applies minimax normalization to the given tensor.

        median_attention_result(attention_dict):
            Computes the median attention result from the attention dictionary.
    """

    @staticmethod
    def median_attention_layers(qkv: dict) -> pd.DataFrame:
        """
        Computes the mean attention values across layers.

        Args:
            qkv (dict):
                The query-key-value pairs for attention layers.

        Returns:
            pd.DataFrame:
                A dataframe containing the mean attention values.
        """
        logger.info("Computing median attention values across layers.")

        result = qkv[[*qkv.keys()][0]]["pair_indices_values"]
        df = pd.DataFrame(dict(result), index=[0]).T.reset_index()
        df.columns = ["pair", "attention"]
        df["pair"] = df["pair"].apply(lambda x: json.loads(x))
        df["pair_first"] = df["pair"].apply(lambda x: x[0])
        df["pair_end"] = df["pair"].apply(lambda x: x[1])
        df = (
            df[["attention", "pair_first"]]
            .groupby("pair_first", as_index=False)
            .agg(
                **{
                    "attention_mean": ("attention", "mean"),
                    "attention": ("attention", "median"),
                }
            )
            .reset_index(drop=True)
        )
        # TODO: 考慮使用不同的統計指標

        max_value = df["attention"].max()
        min_value = df["attention"].min()

        if max_value > min_value:
            df["attention"] = (df["attention"] - min_value) / (max_value - min_value)
        else:
            df["attention"] = df["attention"] - min_value

        return df.round(5)
        # TODO: 考慮到不同block的結果
        # from collections import defaultdict, Counter
        # result = defaultdict(int)
        # total_layer_name = [*qkv.keys()]
        # for layer_name in total_layer_name:
        #     result.update(Counter(qkv[layer_name]["pair_indices_values"]))
        # for pair_name in result.keys():
        #     # result[pair_name] = np.round(result[pair_name] / len(total_layer_name), 3)
        #     result[pair_name] = result[pair_name] / len(total_layer_name)

    @staticmethod
    def _deal_block_matrix(matrix: torch.Tensor) -> torch.Tensor:
        """
        Processes the attention matrix.

        Args:
            matrix (torch.Tensor): The attention matrix.

        Returns:
            torch.Tensor: The processed attention matrix.
        """
        logger.debug(f"Dealing with attention matrix of shape: {matrix.shape}")

        batch_size, input_shape, cache_shape = matrix.shape
        median_value = {}
        if input_shape == cache_shape:
            median_value = matrix.median(dim=2).values.reshape(-1)
        else:
            median_value = matrix.reshape(-1)
        return median_value

    def _retrive_median_attention_dict(self, attention_dict: dict) -> tuple:
        """
        Retrieves the median attention values from the attention dictionary.

        Args:
            attention_dict (dict): Dictionary containing attention values.

        Returns:
            tuple: Processed attention values, number of attention heads, and number of hidden layers.
        """
        logger.debug(
            "Retrieving median attention values from the attention dictionary."
        )

        output_length = attention_dict["output_length"]
        num_attention_heads = attention_dict["num_attention_heads"]
        num_hidden_layers = attention_dict["num_hidden_layers"]
        attention_data = attention_dict["attention"]

        indices = torch.cartesian_prod(
            torch.arange(output_length),
            torch.arange(num_hidden_layers),
            torch.arange(num_attention_heads),
        )
        attention_postprocess = {}
        for idx, value in enumerate(indices):
            output_idx, block_idx, head_idx = value.tolist()
            if idx == 0:
                input_shape = attention_data[output_idx][block_idx].shape[-2]

            median_value = self._deal_block_matrix(
                attention_data[output_idx][block_idx][:, head_idx, :, :]
            )
            if output_idx not in attention_postprocess:
                attention_postprocess[output_idx] = dict()
            if block_idx not in attention_postprocess[output_idx]:
                attention_postprocess[output_idx][block_idx] = dict()
            attention_postprocess[output_idx][block_idx][head_idx] = median_value[
                :input_shape
            ]
        return attention_postprocess, num_attention_heads, num_hidden_layers

    @staticmethod
    def _minimax_normalization(value: torch.Tensor) -> torch.Tensor:
        """
        Applies minimax normalization to the given tensor.

        Args:
            value (torch.Tensor): The tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        logger.debug("Applying minimax normalization.")

        max_value = value.max()
        min_value = value.min()

        if max_value > min_value:
            return ((value - min_value) / (max_value - min_value)).numpy()
        else:
            return (value - min_value).numpy()

    def median_attention_result(self, attention_dict: dict) -> dict:
        """
        Computes the median attention result from the attention dictionary.

        Args:
            attention_dict (dict): Dictionary containing attention values.

        Returns:
            dict: Dictionary containing raw attention data, median attention dataframe,
            and grouped median attention dataframe.
        """
        logger.info("Computing model attention result.")

        (
            attention_postprocess,
            num_attention_heads,
            num_hidden_layers,
        ) = self._retrive_median_attention_dict(attention_dict=attention_dict)
        median_df = pd.json_normalize(attention_postprocess).T.reset_index()
        median_df.columns = ["index_idx", "tensor"]
        median_df[["output_idx", "block_idx", "head_idx"]] = median_df[
            "index_idx"
        ].str.split(".", expand=True)

        median_df["tensor"] = median_df["tensor"].apply(
            lambda x: self._minimax_normalization(x)
        )
        # FIX BUG: https://ppt.cc/fuZ1Cx
        mean_group = (
            median_df.groupby("output_idx")["tensor"]
            .apply(lambda x: np.mean([*x], axis=0))
            .reset_index()
        )
        mean_group["output_idx"] = mean_group["output_idx"].astype(int)
        mean_group.sort_values(by="output_idx", ignore_index=True, inplace=True)

        return {
            "attention_raw_dict": attention_postprocess,
            "median_raw_df": median_df,
            "mean_group_df": mean_group,
        }


class DecoderPostprocessor(object):
    """
    A class for postprocessing decoder outputs from a model.

    Args:
        output (ModelOutput or torch.Tensor):
            The output from the model.
        input_ids (torch.Tensor):
            The input_ids for the model.
        tokenizer (Any):
            The tokenizer associated with the model.

    Attributes:
        output (ModelOutput or torch.Tensor):
            The output from the model.
        input_ids (torch.Tensor):
            The input IDs for the model.
        tokenizer (Any):
            The tokenizer associated with the model.

    Methods:
        find_start_end_position(start_logit, end_logit, offset_mapping, text, max_answer_length=30, n_largest=20,
            *args, **kwargs):
            Finds the start and end positions for an answer span in a QA task.

        _deal_ques_ans_output(output, *args, **kwargs):
            Deals with the output for a QA task.

        _deal_hf_output(output):
            Deals with the Huggingface model output.

        _decode_output(output, tokenizer, *args, **kwargs):
            Decodes the model output.

        _decode_intput(input_ids, tokenizer, *args, **kwargs):
            Decodes the input_ids.

        __call__(*args, **kwargs):
            Calls the DecoderPostprocessor object to decode the model output and input ids.
    """

    def __init__(self, output, input_ids, tokenizer):
        assert not isinstance(
            output, type(None)
        ), "Please provide Tokenizer for decoding output of LLM inference."
        assert isinstance(
            input_ids, torch.Tensor
        ), f"Input type is : {type(input_ids)}. Please change your input type to torch.Tensor."
        assert not isinstance(
            tokenizer, type(None)
        ), "Please provide Tokenizer for decoding tokenizer of LLM inference."

        self.output = output
        self.input_ids = input_ids
        self.tokenizer = tokenizer

    def find_start_end_position(
        self,
        start_logit: torch.Tensor,
        end_logit: torch.Tensor,
        offset_mapping: torch.Tensor,
        text: str,
        max_answer_length: Optional[int] = 30,
        n_largest: Optional[int] = 20,
        *args,
        **kwargs,
    ):
        """
        Finds the start and end positions for an answer span in a QA task.

        Args:
            start_logit (torch.Tensor):
                The start logits.
            end_logit (torch.Tensor):
                The end logits.
            offset_mapping (torch.Tensor):
                The offset mapping of tokens.
            text (str):
                The context text.
            max_answer_length (Optional[int]):
                The maximum length of the answer. Defaults to 30.
            n_largest (Optional[int]):
                The number of largest logits to consider. Defaults to 20.

        Raises:
            AssertionError:
                If the offset_mapping is not provided.

        Returns:
            str:
                The predicted answer span.
        """
        best_score = float("-inf")
        start_indices = (-start_logit).argsort(axis=-1)[0]
        end_indices = (-end_logit).argsort(axis=-1)[0]
        first_word_position = None
        last_word_position = None

        # offset_mapping 是一個列表，列表中的每個元素是一個元組，表示 token 在原始文字中的起始和結束位置
        assert not isinstance(
            offset_mapping, type(None)
        ), "Please give offset_mapping to calcualte loss function in QA tasks."

        for start_idx in start_indices[:n_largest]:
            for end_idx in end_indices[:n_largest]:
                if (
                    offset_mapping[0][start_idx] is None
                    or offset_mapping[0][end_idx] is None
                    or end_idx < start_idx
                    or end_idx - start_idx + 1 > max_answer_length
                ):
                    continue
                score = start_logit[0][start_idx] + end_logit[0][end_idx]
                if score > best_score:
                    best_score = score
                    first_word_position = offset_mapping[0][start_idx][0]
                    last_word_position = offset_mapping[0][end_idx][1]

        if not first_word_position or not last_word_position:
            logger.warning("No answer found.")
            return ""

        return text[first_word_position:last_word_position]

    def _deal_ques_ans_output(self, output, *args, **kwargs):
        """
        Deals with the output for a QA task.

        Args:
            output (ModelOutput or torch.Tensor):
                The output from the model.

        Raises:
            AssertionError:
                If the offset_mapping or text is not provided.

        Returns:
            str:
                The predicted answer span.
        """
        if isinstance(output, ModelOutput):
            text = kwargs.pop("text", None)
            offset_mapping = kwargs.pop("offset_mapping", None)
            assert not isinstance(
                offset_mapping, type(None)
            ), "Please provide offset_mapping."
            assert text, "Please provide Question context."
            predict = self.find_start_end_position(
                start_logit=output["start_logits"],
                end_logit=output["end_logits"],
                offset_mapping=offset_mapping,
                text=text,
                *args,
                **kwargs,
            )
        elif isinstance(output, torch.Tensor):
            text = kwargs.pop("text", None)
            assert kwargs.get(
                "offset_mapping", None
            ), "Please provide tokens offset_mapping."
            assert text, "Please provide question context."
            assert kwargs.get("start_logits", None), "Please provide start_logits."
            assert kwargs.get("end_logits", None), "Please provide end_logits."

            predict = self.find_start_end_position(
                start_logit=kwargs["start_logits"],
                end_logit=kwargs["end_logits"],
                offset_mapping=kwargs["offset_mapping"],
                *args,
                **kwargs,
            )
        else:
            ValueError(
                "Please check type of output is torch.Tensor or ModeulOutput(Huggingface)."
            )
        return predict, None

    def _deal_hf_output(self, output):
        """
        Deals with the Huggingface model output.

        Args:
            output (ModelOutput or torch.Tensor):
                The output from the model.

        Returns:
            torch.Tensor:
                The processed output.
        """
        if "sequences" in output:
            output = output.sequences[0]
        elif "logits" in output:
            output = output.logits.argmax(dim=-1)[0]
        else:
            ValueError(
                "Please check output have `sequences` or `logits` to decode model predict."
            )
        return output

    def _decode_output(self, output, tokenizer, *args, **kwargs):
        """
        Decodes the model output.

        Args:
            output (ModelOutput or torch.Tensor):
                The output from the model.
            tokenizer (Any):
                The tokenizer associated with the model.

        Raises:
            ValueError:
                If the output type is not ModelOutput or torch.Tensor.

        Returns:
            str:
                The decoded output.
        """
        logger.debug(f"Decoding model output with type: {type(output)}")
        # Question-Answering task
        if kwargs.get("task_type", None) == "question_answering":
            return self._deal_ques_ans_output(output, *args, **kwargs)

        # Non Question-Answering task
        if isinstance(output, ModelOutput):
            output = self._deal_hf_output(output)

        elif isinstance(output, torch.Tensor):
            assert (
                len(output.shape) == 1
            ), f"The output dimension {output.shape} should be checked before decoding."
            pass

        else:
            ValueError(
                "This toolbox currently supports only Huggingface output and torch.tensor."
            )

        output_decode = []
        input_shape = kwargs.get("input_shape", None)
        if input_shape is not None:
            for text in output[input_shape:].cpu():
                output_decode.append(tokenizer.decode(text))
        else:
            for text in output.cpu():
                output_decode.append(tokenizer.decode(text))

        predict = "".join(output_decode)
        return predict, output_decode

    def _decode_intput(self, input_ids, tokenizer, *args, **kwargs):
        """
        Decodes the input IDs.

        Args:
            input_ids (torch.Tensor):
                The input IDs for the model.
            tokenizer (Any):
                The tokenizer associated with the model.

        Raises:
            AssertionError:
                If the input dimension is not two.

        Returns:
            list:
                The decoded input.
        """
        logger.debug(f"Decoding model input with type: {type(input_ids)}")

        assert (
            len(input_ids.shape) == 2
        ), f"The input dimension is {input_ids.shape}, currenly support \
        two-dimension element = (1 batch_size, tokens_number)."
        # input decode
        prompt_decode = []
        for text in input_ids.cpu()[0]:
            prompt_decode.append(tokenizer.decode(text))

        return prompt_decode

    def __call__(self, *args, **kwargs):
        predict_decode, output_decode = self._decode_output(
            self.output, self.tokenizer, *args, **kwargs
        )
        prompt_decode = self._decode_intput(
            self.input_ids, self.tokenizer, *args, **kwargs
        )

        return predict_decode, output_decode, prompt_decode
