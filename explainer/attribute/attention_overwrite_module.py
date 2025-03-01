import importlib
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List


# Copy from transformers.modeling_`specific_model`.py
# We refer to repos of daam, BertViz and Huggingface,
# thanks for their wonderful work in Explainable AI and NLP field.

# TODO: Currently we only support the model with rotary positional embeddings.
# We will support more models in the future.


class AttentionHandler:
    """
    A class to handle attention operations.

    Args:
        apply_rotary_pos_emb (callable): Function to apply rotary positional embeddings.
        repeat_kv (callable): Function to repeat key or value tensors.
        num_attention_heads (int): Number of attention heads.
        num_hidden_layers (int): Number of hidden layers.

    Returns:
        AttentionHandler: The attention handler object.

    Constraints:
        Currently only supports models with rotary positional embeddings.
    """

    def __init__(
        self,
        apply_rotary_pos_emb=None,
        repeat_kv=None,
        num_attention_heads=None,
        num_hidden_layers=None,
    ):
        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        self.repeat_kv = repeat_kv
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self._query = []
        self._key = []
        self._values = []
        self.attn_output = []
        self.key_or_value_toggle = 0

    @staticmethod
    def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Clones a tensor and moves it to CPU.

        Args:
            tensor (torch.Tensor): Tensor to be cloned.

        Returns:
            torch.Tensor: Cloned tensor.
        """
        return tensor.clone().cpu().detach()

    def _list_reshape(self, tensor) -> List:
        """
        Splits the tensor into chunks based on the number of hidden layers.

        Args:
            tensor (List[torch.Tensor]): List of tensors to be split.

        Returns:
            List[List[torch.Tensor]]: List of lists of tensors, split by hidden layers,
            the return shape in list should be (batch, num_attention_heads, seqlen, head_dim),
            the list length should be the same as the number of output.
        """
        self.output_len = len(tensor) // self.num_hidden_layers
        return [
            tensor[i * self.num_hidden_layers: (i + 1) * self.num_hidden_layers]
            for i in range(self.output_len)
        ]

    def _reshape_queries(self, tensor) -> List:
        """
        Reshapes queries for further processing.

        Args:
            tensor (List[List[torch.Tensor]]): List of lists of query tensors.

        Returns:
            List[torch.Tensor]: Reshaped queries.
        """
        return [torch.cat(i, dim=2) for i in zip(*tensor)]

    def get_query(
        self, q, k, cos, sin, position_ids, unsqueeze_dim=1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary positional embeddings to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            cos (torch.Tensor): Cosine component for rotary embeddings.
            sin (torch.Tensor): Sine component for rotary embeddings.
            position_ids (torch.Tensor): Position IDs tensor.
            unsqueeze_dim (int, optional): Dimension to unsqueeze. Default is 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the modified query and key tensors.
        """
        query_states, key_states = self.apply_rotary_pos_emb(
            q, k, cos, sin, position_ids, unsqueeze_dim=1
        )
        self._query.append(self._clone_tensor(query_states))
        return query_states, key_states

    def get_key_or_values(
        self, hidden_states: torch.Tensor, n_rep: int
    ) -> torch.Tensor:
        """
        Repeats key or value tensors based on the toggle state.

        Args:
            hidden_states (torch.Tensor): Hidden states tensor.
            n_rep (int): Number of repetitions.

        Returns:
            torch.Tensor: Repeated key or value tensor.
        """
        if self.key_or_value_toggle == 0:
            key_states = self.repeat_kv(hidden_states, n_rep)
            self._key.append(self._clone_tensor(key_states))
            self.key_or_value_toggle = 1
            return key_states
        else:
            value_states = self.repeat_kv(hidden_states, n_rep)
            self._values.append(self._clone_tensor(value_states))
            self.key_or_value_toggle = 0
            return value_states

    def retrieve_store_qkv(self) -> Tuple[list, list, list]:
        """
        Retrieves stored query, key, and value tensors.

        Returns:
            Tuple containing lists of stored query, key, and value tensors.
        """
        return (
            self._query.copy(),
            self._key.copy(),
            self._values.copy(),
        )

    def calculate_attention(self, query: list, key: list) -> list:
        """
        Calculates attention weights.

        Args:
            queries (List[torch.Tensor]): List of query tensors.
            keys (List[torch.Tensor]): List of key tensors.

        Returns:
            List[torch.Tensor]: List of attention weights tensors.
        """
        # BUG: https://github.com/pytorch/pytorch/issues/101359
        for q, k in zip(query, key):
            attn_weights = torch.matmul(
                q.float(), k.float().transpose(2, 3)
            ) / math.sqrt(self.num_attention_heads)
            # batch_size, num_heads, _, seq_len = attn_weights.size()
            self.attn_output.append(attn_weights)
        return self.attn_output.copy()

    def retrieve_attention_value(self) -> Tuple[list, list, list, list]:
        """
        Retrieves attention values.

        Returns:
            Dict: containing lists of lists of query, key, value, and
            attention tensors split by hidden layers.
        """
        queries, keys, values = self.retrieve_store_qkv()
        attentions = self.calculate_attention(query=queries, key=keys)
        reshaped_queries = self._list_reshape(queries)

        return {
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "output_length": self.output_len,
            "raw_query": reshaped_queries,
            "query": self._reshape_queries(reshaped_queries),
            "keys": self._list_reshape(keys),
            "values": self._list_reshape(values),
            "attention": self._list_reshape(attentions),
        }


def overwrite_module(
    model: nn.Module,
    num_attention_heads: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
):
    """
    Overwrites the model's module functions with custom implementations.

    Args:
        model (nn.Module): Model to overwrite.
        num_attention_heads (int, optional): Number of attention heads. Defaults to model's configuration.
        num_hidden_layers (int, optional): Number of hidden layers. Defaults to model's configuration.

    Returns:
        Tuple[nn.Module, callable]: Tuple containing the modified module and a callable to retrieve attention values.
    """
    module = importlib.import_module(f"{model.__module__}")
    apply_rotary_pos_emb = module.apply_rotary_pos_emb
    repeat_kv = module.repeat_kv
    num_attention_heads = (
        model.config.num_attention_heads
        if num_attention_heads is None
        else num_attention_heads
    )
    num_hidden_layers = (
        model.config.num_hidden_layers
        if num_hidden_layers is None
        else num_hidden_layers
    )
    attention_handler = AttentionHandler(
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        repeat_kv=repeat_kv,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
    )
    module.apply_rotary_pos_emb = attention_handler.get_query
    module.repeat_kv = attention_handler.get_key_or_values
    return module, attention_handler.retrieve_attention_value
