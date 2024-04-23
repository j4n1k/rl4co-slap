from typing import Optional, Union

import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.base import ConstructivePolicy
from rl4co.models.common.constructive.nonautoregressive.decoder import (
    NonAutoregressiveDecoder,
)
from rl4co.models.common.constructive.nonautoregressive.encoder import (
    NonAutoregressiveEncoder,
)
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class NonAutoregressivePolicy(ConstructivePolicy):

    """
    # TODO
    Base Non-autoregressive policy for NCO construction methods.
    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (non-autoregressively) to construct the solution to the NCO problem


    Warning:
        The effectiveness of the non-autoregressive approach can vary significantly across different problem types and configurations.
        It may require careful tuning of the model architecture and decoding strategy to achieve competitive results.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        embed_dim: Dimension of the embeddings
        num_graph_layers: Number of layers in the encoder
        num_decoder_layers: Number of layers in the decoder # TODO
        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **constructive_policy_kw: Unused keyword arguments
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        embed_dim: int = 64,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn="silu",
        agg_fn="mean",
        linear_bias: bool = True,
        train_decode_type: str = "multistart_sampling",
        val_decode_type: str = "multistart_greedy",
        test_decode_type: str = "multistart_greedy",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")
            if "decoder" in constructive_policy_kw:
                raise ValueError(
                    "NonAutoregressivePolicy does not accept 'decoder' as a keyword argument. "
                    "The decoder is fixed to the heatmap decoder."
                )

        if encoder is None:
            log.info("Initializing default NonAutoregressiveEncoder")
            encoder = NonAutoregressiveEncoder(
                embed_dim=embed_dim,
                env_name=env_name,
                init_embedding=init_embedding,
                edge_embedding=edge_embedding,
                graph_network=graph_network,
                heatmap_generator=heatmap_generator,
                num_layers_heatmap_generator=num_layers_heatmap_generator,
                num_layers_graph_encoder=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
                linear_bias=linear_bias,
            )

        # The decoder generates logits given the current td and heatmap
        decoder = NonAutoregressiveDecoder()

        # Pass to constructive policy
        super(NonAutoregressivePolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )
