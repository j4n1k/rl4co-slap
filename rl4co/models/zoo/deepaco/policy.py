from functools import partial
from typing import Optional, Type, Union

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.deepaco.antsystem import AntSystem
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.utils import merge_with_defaults
from rl4co.utils.ops import batchify


class DeepACOPolicy(NonAutoregressivePolicy):
    """Implememts DeepACO policy based on :class:`NonAutoregressivePolicy`. Introduced by Ye et al. (2023): https://arxiv.org/abs/2309.14032.
    This policy uses a Non-Autoregressive Graph Neural Network to generate heatmaps,
    which are then used to run Ant Colony Optimization (ACO) to construct solutions.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        env_name: Name of the environment used to initialize embeddings
        aco_class: Class representing the ACO algorithm to be used. Defaults to :class:`AntSystem`.
        aco_kwargs: Additional arguments to be passed to the ACO algorithm.
        n_ants: Number of ants to be used in the ACO algorithm. Can be an integer or dictionary. Defaults to 20.
        n_iterations: Number of iterations to run the ACO algorithm. Can be an integer or dictionary. Defaults to `dict(train=1, val=20, test=100)`.
        ls_reward_aug_W: Coefficient to be used for the reward augmentation with the local search. Defaults to 0.95.
        encoder_kwargs: Additional arguments to be passed to the encoder.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        env_name: str = "tsp",
        aco_class: Optional[Type[AntSystem]] = None,
        aco_kwargs: dict = {},
        n_ants: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        ls_reward_aug_W: float = 0.95,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = NARGNNEncoder(**encoder_kwargs)

        super(DeepACOPolicy, self).__init__(
            encoder=encoder,
            env_name=env_name,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )

        self.aco_class = AntSystem if aco_class is None else aco_class
        self.aco_kwargs = aco_kwargs
        self.n_ants = merge_with_defaults(n_ants, train=30, val=48, test=48)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=5, test=10)
        self.ls_reward_aug_W = ls_reward_aug_W

    def forward(
        self,
        td_initial: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        calc_reward: bool = True,
        phase: str = "train",
        actions=None,
        return_actions: bool = True,
        return_hidden: bool = True,
        **kwargs,
    ):
        """
        Forward method. During validation and testing, the policy runs the ACO algorithm to construct solutions.
        See :class:`NonAutoregressivePolicy` for more details during the training phase.
        """
        n_ants = self.n_ants[phase]
        # Instantiate environment if needed
        if (phase != "train" or self.ls_reward_aug_W > 0) and (env is None or isinstance(env, str)):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        if phase == "train":
            select_start_nodes_fn = partial(
                self.aco_class.select_start_node_fn, start_node=self.aco_kwargs.get("start_node", None)
            )
            kwargs.update({"select_start_nodes_fn": select_start_nodes_fn})
            #  we just use the constructive policy
            outdict = super().forward(
                td_initial,
                env,
                phase=phase,
                decode_type="multistart_sampling",
                calc_reward=calc_reward,
                num_starts=n_ants,
                actions=actions,
                return_actions=return_actions,
                return_hidden=return_hidden,
                **kwargs,
            )

            if self.ls_reward_aug_W > 0 and self.aco_kwargs.get("use_local_search", False):
                heatmap_logits = outdict["hidden"]
                aco = self.aco_class(heatmap_logits, n_ants=n_ants, **self.aco_kwargs)
                
                actions = outdict["actions"]
                _, ls_reward = aco.local_search(batchify(td_initial, n_ants), env, actions)
                outdict["reward"] = outdict["reward"] * (1 - self.ls_reward_aug_W) + ls_reward * self.ls_reward_aug_W

            return outdict

        heatmap_logits, _ = self.encoder(td_initial)

        aco = self.aco_class(heatmap_logits, n_ants=n_ants, **self.aco_kwargs)
        td, actions, reward = aco.run(td_initial, env, self.n_iterations[phase])

        if calc_reward:
            td.set("reward", reward)

        out = {"reward": td["reward"]}

        if return_actions:
            out["actions"] = actions

        return out
