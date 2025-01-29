from typing import Optional

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch import nn
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs import get_env
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.models import AttentionModelPolicy, ConstructivePolicy, ConstructiveEncoder, ConstructiveDecoder, \
    AutoregressiveEncoder, AutoregressiveDecoder, MatNetPolicy
from rl4co.models.common.constructive.base import NoEncoder
from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.env_embeddings.init import MTSPInitEmbedding
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.utils import RL4COTrainer
from rl4co.utils.decoding import DecodingStrategy, get_decoding_strategy, get_log_likelihood, process_logits
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length, calculate_entropy
from rl4co.models.zoo import AttentionModel

from typing import Callable, Union

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    def discrete_cmap(num, base_cmap="nipy_spectral"):
        """Create an N-bin discrete colormap from the specified input map"""
        base = colormaps[base_cmap]
        color_list = base(np.linspace(0, 1, num))
        cmap_name = base.name + str(num)
        return base.from_list(cmap_name, color_list, num)

    if actions is None:
        actions = td.get("action", None)
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    num_agents = td["num_agents"]
    locs = td["locs"]
    cmap = discrete_cmap(num_agents, "rainbow")
    fig, ax = plt.subplots()

    # Add depot action = 0 to before first action and after last action
    actions = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64),
            actions,
            torch.zeros(1, dtype=torch.int64),
        ]
    )

    # Make list of colors from matplotlib
    for i, loc in enumerate(locs):
        if i == 0:
            # depot
            marker = "s"
            color = "g"
            label = "Depot"
            markersize = 10
        elif i == 1:
            # cs
            marker = "s"
            color = "r"
            label = "CS"
            markersize = 10
        else:
            # normal location
            marker = "o"
            color = "tab:blue"
            label = "Customers"
            markersize = 8
        if i > 1:
            label = ""

        ax.plot(
            loc[0],
            loc[1],
            color=color,
            marker=marker,
            markersize=markersize,
            label=label,
        )

    # Plot the actions in order
    agent_idx = 0
    agent_travel_time = 0
    max_travel = 0
    n_charges = {i+1: 0 for i in range(num_agents)}
    battery_levels = {i+1: [100] for i in range(num_agents)}
    for i in range(len(actions)):
        if actions[i] == 0:
            agent_idx += 1
            if agent_travel_time > max_travel:
                max_travel = agent_travel_time
            agent_travel_time = 0
        color = cmap(num_agents - agent_idx)

        from_node = actions[i]
        to_node = (
            actions[i + 1] if i < len(actions) - 1 else actions[0]
        )  # last goes back to depot
        from_loc = td["locs"][from_node]
        to_loc = td["locs"][to_node]
        travel_time = get_distance(from_loc, to_loc)
        if agent_idx <= num_agents:
            battery_levels[agent_idx].append(battery_levels[agent_idx][-1] - 2 * travel_time)
        if to_node == 1:
            travel_time += 300
            n_charges[agent_idx] += 1
            battery_levels[agent_idx].append(100)
        agent_travel_time += travel_time
        ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color)
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="->", color=color),
            annotation_clip=False,
        )
    print(max_travel)
    print(n_charges)
    print(battery_levels)
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("mTSP")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    return fig



class MTSPGenerator(Generator):
    """Data generator for the Multiple Travelling Salesman Problem (mTSP).

    Args:
        num_loc: number of locations (customers) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        min_num_agents: minimum number of agents (vehicles), include
        max_num_agents: maximum number of agents (vehicles), include

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            num_agents [batch_size]: number of agents (vehicles)
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        min_num_agents: int = 5,
        max_num_agents: int = 5,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample the number of agents
        num_agents = torch.randint(
            self.min_num_agents,
            self.max_num_agents + 1,
            size=(*batch_size,),
        )

        return TensorDict(
            {
                "locs": locs,
                "num_agents": num_agents,
            },
            batch_size=batch_size,
        )


class MTSPEnv(RL4COEnvBase):
    """Multiple Traveling Salesman Problem environment
    At each step, an agent chooses to visit a city. A maximum of `num_agents` agents can be employed to visit the cities.
    The cost can be defined in two ways:
        - `minmax`: (default) the reward is the maximum of the path lengths of all the agents
        - `sum`: the cost is the sum of the path lengths of all the agents
    Reward is - cost, so the goal is to maximize the reward (minimize the cost).

    Observations:
        - locations of the depot and each customer.
        - number of agents.
        - the current agent index.
        - the current location of the vehicle.

    Constrains:
        - each agent's tour starts and ends at the depot.
        - each customer must be visited exactly once.

    Finish condition:
        - all customers are visited and all agents back to the depot.

    Reward:
        There are two ways to calculate the cost (-reward):
        - `minmax`: (default) the cost is the maximum of the path lengths of all the agents.
        - `sum`: the cost is the sum of the path lengths of all the agents.

    Args:
        cost_type: type of cost to use, either `minmax` or `sum`
        generator: MTSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "mtsp"

    def __init__(
        self,
        generator: MTSPGenerator = None,
        generator_params: dict = {},
        cost_type: str = "minmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MTSPGenerator(**generator_params)
        self.generator = generator
        self.cost_type = cost_type
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        # Make actionspace multi dimensional? -> [[Locations], [charging_duration]]
        dist_mat = td["dist_mat"]
        # Initial variables
        is_first_action = batch_to_scalar(td["i"]) == 0
        last_node_to_all = gather_by_index(dist_mat, idx=td["current_node"])
        # distance traveled from last node to current node
        dist_traveled = gather_by_index(last_node_to_all, idx=td["action"])
        target_battery_level = gather_by_index(td["thresholds"], td["charging_duration"])
        td["battery"] -= 10 * dist_traveled.view(td["battery"].shape)
        cs_mask = td["action"] == 1
        # if traveled to charging station, recharge
        td["battery"][cs_mask] = target_battery_level[cs_mask]
        current_node = td["action"]
        first_node = current_node if is_first_action else td["first_node"]

        # Get the locations of the current node and the previous node and the depot
        cur_loc = gather_by_index(td["locs"], current_node)
        prev_loc = gather_by_index(
            td["locs"], td["current_node"]
        )  # current_node is the previous node
        depot_loc = td["locs"][..., 0, :]

        # If current_node is the depot, then increment agent_idx
        cur_agent_idx = td["agent_idx"] + (current_node == 0).long()
        # If agent_idx is increased, vehicle is refueled
        td["battery"] += (current_node == 0).long().view(td["battery"].shape) * (100 - td["battery"])
        comp = td["thresholds"] >= td["battery"][:, None, :]
        comp = comp.squeeze()

        # Set not visited to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node[..., None].expand_as(td["available"]), 0
        )
        # Available[..., 0] is the depot, which is always available unless:
        # - current_node is the depot
        # - agent_idx greater than num_agents -1
        available[..., 0] = torch.logical_and(
            current_node != 0, td["agent_idx"] < td["num_agents"] - 1
        )

        current_node_to_all = gather_by_index(dist_mat, idx=current_node)
        proj_depletion = current_node_to_all * 10
        proj_battery = td["battery"] - proj_depletion
        reachable = proj_battery > td["threshold"]

        battery_check = td["battery"] < td["threshold"]

        # We are done there are no unvisited locations except the depot
        done = torch.count_nonzero(available[..., 1:], dim=-1) == 0

        # If done is True, then we make the depot available again, so that it will be selected as the next node with prob 1
        available[..., 0] = torch.logical_or(done, available[..., 0])
        if done.any():
            pass

        if battery_check.any():
            # battery below threshold, needs to be recharged
            charging_mask = available.clone()
            idx = torch.squeeze(battery_check).nonzero().squeeze(0)
            charging_mask[idx] = td["charging_mask"][idx]
            action_mask = charging_mask
        else:
            # action_mask = available
            action_mask = available & reachable
            # make charging station visitable
            action_mask[..., 1] = 1
            if done.any():
                idx = torch.squeeze(done).nonzero().squeeze(0)
                action_mask[idx, 1] = 0
                action_mask[idx, 0] = 1
            # just_charged = td["battery"] == 100
            # make sure that charging location can not be visited if just recharged
            if cs_mask.any():
                idx = torch.squeeze(cs_mask).nonzero().squeeze(0)
                action_mask[idx, 1] = 0
        for i in range(action_mask.shape[0]):
            if torch.all(action_mask[i] == False):
               # print("ERROR")
                # TODO: Vehicle was just charged but to low and we can't reach anything else -> need to recharge again
                # Edge Case
                action_mask[i, 1] = 1
        # Update the current length
        current_length = td["current_length"] + get_distance(cur_loc, prev_loc)

        # Update service time
        current_duration = td["current_duration"] + get_distance(cur_loc, prev_loc)
        current_duration[cs_mask] += 5

        # If done, we add the distance and duration from the current_node to the depot as well
        current_length = torch.where(
            done, current_length + get_distance(cur_loc, depot_loc), current_length
        )

        current_duration = torch.where(
            done, current_duration + get_distance(cur_loc, depot_loc), current_duration
        )

        # We update the max_subtour_length and reset the current_length
        max_subtour_length = torch.where(
            current_length > td["max_subtour_length"],
            current_length,
            td["max_subtour_length"],
        )

        max_subtour_duration = torch.where(
            current_duration > td["max_subtour_duration"],
            current_duration,
            td["max_subtour_duration"],
        )

        # If current agent is different from previous agent, then we have a new subtour and reset the length
        current_length *= (cur_agent_idx == td["agent_idx"]).float()
        current_duration *= (cur_agent_idx == td["agent_idx"]).float()

        # The reward is the negative of the max_subtour_length (minmax objective)
        # reward = -max_subtour_length
        reward = -max_subtour_duration
        if done.all():
            pass
        td.update(
            {
                "max_subtour_length": max_subtour_length,
                "max_subtour_duration": max_subtour_duration,
                "current_length": current_length,
                "current_duration": current_duration,
                "agent_idx": cur_agent_idx,
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "charging_duration_mask": comp,
                "reward": reward,
                "done": done,
                "available": available
            }
        )

        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        # Keep track of the agent number to know when to stop
        agent_idx = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        # Make variable for max_subtour_length between subtours
        max_subtour_length = torch.zeros(batch_size, dtype=torch.float32, device=device)
        current_length = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # service time tracker
        max_subtour_duration = torch.zeros(batch_size, dtype=torch.float32, device=device)
        current_duration = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # Other variables
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.generator.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        available[..., 0] = 0  # Depot is not available as first node
        available[..., 1] = 0  # charging station is not available as first node
        charging_mask = ~available
        charging_mask[..., 0] = 0
        # Battery constraints
        battery = torch.full((*batch_size, 1), 100, dtype=torch.float32)
        threshold = torch.full(
            (*batch_size, 1), 30, device=device
        )
        values = torch.tensor([100.0])
        n_th = values.shape[0]
        charging_duration_mask = torch.ones((*batch_size, n_th), dtype=torch.bool, device=device)
        # thresholds = torch.full((batch_size[0], 5, 1), 70.0)
        thresholds = torch.arange(
            values.min().item(), values.max().item() + 1, step=10, device=device).repeat(
            batch_size[0], 1).view(batch_size[0], n_th, 1)
        charging_duration = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        dist_mat = self._get_distance_matrix(td["locs"])
        dist_mat = dist_mat.to(device)
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": td["locs"],  # depot is first node
                "num_agents": td["num_agents"],
                "max_subtour_length": max_subtour_length,
                "max_subtour_duration": max_subtour_duration,
                "current_length": current_length,
                "current_duration": current_duration,
                "agent_idx": agent_idx,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "available": available,
                "action_mask": available,
                "charging_mask": charging_mask,
                "charging_duration_mask": charging_duration_mask,
                "charging_duration": charging_duration,
                "dist_mat": dist_mat,
                "battery": battery,
                "threshold": threshold,
                "thresholds": thresholds
            },
            batch_size=batch_size,
        )

    @staticmethod
    def _get_distance_matrix(locs: torch.Tensor):
        """Compute the Manhattan distance matrix for the given coordinates.

        Args:
            locs: Tensor of shape [..., n, dim]
        """
        if locs.dtype != torch.float32 and locs.dtype != torch.float64:
            locs = locs.to(torch.float32)

            # Compute pairwise differences
        diff = locs[..., :, None, :] - locs[..., None, :, :]

        # Compute Manhattan distance
        distance_matrix = torch.sum(torch.abs(diff), dim=-1)
        return distance_matrix

    def _make_spec(self, generator: MTSPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            num_agents=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            agent_idx=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            max_subtour_length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec()
        self.done_spec = UnboundedDiscreteTensorSpec(dtype=torch.bool)

    def _get_reward(self, td, actions=None) -> TensorDict:
        # With minmax, get the maximum distance among subtours, calculated in the model
        if self.cost_type == "minmax":
            return td["reward"].squeeze(-1)

        # With distance, same as TSP
        elif self.cost_type == "sum":
            locs = td["locs"]
            locs_ordered = locs.gather(1, actions.unsqueeze(-1).expand_as(locs))
            return -get_tour_length(locs_ordered)

        else:
            raise ValueError(f"Cost type {self.cost_type} not supported")

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        assert True, "Not implemented"

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)


class ChargingInitEmbedding(nn.Module):
    """Initial embedding for the Multiple Traveling Salesman Problem (mTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, cities)
    """

    def __init__(self, embed_dim, linear_bias=True):
        """NOTE: new made by Fede. May need to be checked"""
        super(ChargingInitEmbedding, self).__init__()
        node_dim = 1  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        threshold_embedding = self.init_embed(td["thresholds"])
        return threshold_embedding


class MTSPContext(EnvContext):
    """Context embedding for the Multiple Traveling Salesman Problem (mTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining_agents
        - current_length
        - max_subtour_length
        - distance_from_depot
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(MTSPContext, self).__init__(embed_dim, 2 * embed_dim)
        proj_in_dim = (
            6  # remaining_agents, current_length, max_subtour_length, distance_from_depot, battery level, dist_to_cs
        )
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding.squeeze()

    def _state_embedding(self, embeddings, td):
        dynamic_feats = torch.stack(
            [
                (td["num_agents"] - td["agent_idx"]).float(),
                td["current_length"],
                td["max_subtour_length"],
                self._distance_from_depot(td),
                td["battery"].view(td["current_length"].shape),
                self._distance_from_cs(td)
            ],
            dim=-1,
        )
        return self.proj_dynamic_feats(dynamic_feats)

    def _distance_from_depot(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 0, :], dim=-1)

    def _distance_from_cs(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 1, :], dim=-1)

    def forward(self, embeddings, td):
        if embeddings.shape[-2] == td["locs"].shape[-2]:
            state_embedding = self._state_embedding(embeddings, td)
            cur_node_embedding = self._cur_node_embedding(embeddings, td)
            context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
            return self.project_context(context_embedding)
        else:
            return embeddings.new_zeros(embeddings.size(0), self.embed_dim)


class MultiConstructivePolicy(nn.Module):
    """
    Base class for constructive policies. Constructive policies take as input and instance and output a solution (sequence of actions).
    "Constructive" means that a solution is created from scratch by the model.

    The structure follows roughly the following steps:
        1. Create a hidden state from the encoder
        2. Initialize decoding strategy (such as greedy, sampling, etc.)
        3. Decode the action given the hidden state and the environment state at the current step
        4. Update the environment state with the action. Repeat 3-4 until all sequences are done
        5. Obtain log likelihood, rewards etc.

    Note that an encoder is not strictly needed (see :class:`NoEncoder`).). A decoder however is always needed either in the form of a
    network or a function.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward may not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        encoder: Encoder to use
        decoder: Decoder to use
        env_name: Environment name to solve (used for automatically instantiating networks)
        temperature: Temperature for the softmax during decoding
        tanh_clipping: Clipping value for the tanh activation (see Bello et al. 2016) during decoding
        mask_logits: Whether to mask the logits or not during decoding
        train_decode_type: Decoding strategy for training
        val_decode_type: Decoding strategy for validation
        test_decode_type: Decoding strategy for testing
    """

    def __init__(
        self,
        location_encoder: Union[ConstructiveEncoder, Callable],
        charging_encoder: Union[ConstructiveEncoder, Callable],
        location_decoder: Union[ConstructiveDecoder, Callable],
        charging_decoder: Union[ConstructiveDecoder, Callable],
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(MultiConstructivePolicy, self).__init__()

        if len(unused_kw) > 0:
            log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.env_name = env_name

        # Encoder and decoder
        # if encoder is None:
        #     log.warning("`None` was provided as encoder. Using `NoEncoder`.")
        #     encoder = NoEncoder()
        self.location_encoder = location_encoder
        self.charging_encoder = charging_encoder
        self.location_decoder = location_decoder
        self.charging_decoder = charging_decoder
        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden_l, init_embeds_location = self.location_encoder(td)
        hidden_c, init_embeds_charging = self.charging_encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy_nodes: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            key="action",
            **decoding_kwargs,
        )

        decode_strategy_charging: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            key="charging_duration",
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy_nodes.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden_locations = self.location_decoder.pre_decoder_hook(td, env, hidden_l, num_starts)

        td, env, num_starts = decode_strategy_charging.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden_charging = self.charging_decoder.pre_decoder_hook(td, env, hidden_c, num_starts)
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.location_decoder(td, hidden_locations, num_starts)
            td = decode_strategy_nodes.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            logits, mask = self.charging_decoder(td, hidden_charging, num_starts)
            td = decode_strategy_charging.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy_nodes.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden_l
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds_location

        return outdict


class MultiAutoregressivePolicy(MultiConstructivePolicy):
    """Template class for an autoregressive policy, simple wrapper around
    :class:`rl4co.models.common.constructive.base.ConstructivePolicy`.

    Note:
        While a decoder is required, an encoder is optional and will be initialized to
        :class:`rl4co.models.common.constructive.autoregressive.encoder.NoEncoder`.
        This can be used in decoder-only models in which at each step actions do not depend on
        previously encoded states.
    """

    def __init__(
        self,
        location_encoder: AutoregressiveEncoder,
        charging_encoder: AutoregressiveEncoder,
        location_decoder: AutoregressiveDecoder,
        charging_decoder: AutoregressiveDecoder,
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "multisampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):

        super(MultiAutoregressivePolicy, self).__init__(
            location_encoder=location_encoder,
            charging_encoder=charging_encoder,
            location_decoder=location_decoder,
            charging_decoder=charging_decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kw,
        )


class MultiAttentionModelPolicy(MultiAutoregressivePolicy):
    """
    Attention Model Policy based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    This model first encodes the input graph using a Graph Attention Network (GAT) (:class:`AttentionModelEncoder`)
    and then decodes the solution using a pointer network (:class:`AttentionModelDecoder`). Cache is used to store the
    embeddings of the nodes to be used by the decoder to save computation.
    See :class:`rl4co.models.common.constructive.autoregressive.policy.AutoregressivePolicy` for more details on the inference process.

    Args:
        encoder: Encoder module, defaults to :class:`AttentionModelEncoder`
        decoder: Decoder module, defaults to :class:`AttentionModelDecoder`
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        env_name: Name of the environment used to initialize embeddings
        encoder_network: Network to use for the encoder
        init_embedding: Module to use for the initialization of the embeddings
        context_embedding: Module to use for the context embedding
        dynamic_embedding: Module to use for the dynamic embedding
        use_graph_context: Whether to use the graph context
        linear_bias_decoder: Whether to use a bias in the linear layer of the decoder
        sdpa_fn_encoder: Function to use for the scaled dot product attention in the encoder
        sdpa_fn_decoder: Function to use for the scaled dot product attention in the decoder
        sdpa_fn: (deprecated) Function to use for the scaled dot product attention
        mask_inner: Whether to mask the inner product
        out_bias_pointer_attn: Whether to use a bias in the pointer attention
        check_nan: Whether to check for nan values during decoding
        temperature: Temperature for the softmax
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        mask_logits: Whether to mask the logits during decoding
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
        moe_kwargs: Keyword arguments for MoE,
            e.g., {"encoder": {"hidden_act": "ReLU", "num_experts": 4, "k": 2, "noisy_gating": True},
                   "decoder": {"light_version": True, ...}}
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: str = "tsp",
        encoder_network: nn.Module = None,
        init_embedding_location: nn.Module = None,
        init_embedding_charging: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "multisampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        moe_kwargs: dict = {"encoder": None, "decoder": None},
        **unused_kwargs,
    ):
        location_encoder = AttentionModelEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            env_name=env_name,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            net=encoder_network,
            init_embedding=init_embedding_location,
            sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
            moe_kwargs=moe_kwargs["encoder"],
        )

        charging_encoder = AttentionModelEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            env_name=env_name,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            net=encoder_network,
            init_embedding=init_embedding_charging,
            sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
            moe_kwargs=moe_kwargs["encoder"],
        )

        location_decoder = AttentionModelDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            key="action_mask",
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias_decoder,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
            moe_kwargs=moe_kwargs["decoder"],
        )

        charging_decoder = AttentionModelDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            key="charging_duration_mask",
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias_decoder,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
            moe_kwargs=moe_kwargs["decoder"],
        )

        super(MultiAttentionModelPolicy, self).__init__(
            location_encoder=location_encoder,
            charging_encoder=charging_encoder,
            location_decoder=location_decoder,
            charging_decoder=charging_decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
# td["test"] = torch.zeros((4,2))
# td["test"][..., 0] = torch.ones(4,1)


def main():
    env = MTSPEnv()
    emb_dim = 128
    # policy = AttentionModelPolicy(env_name=env.name,
    #                               # this is actually not needed since we are initializing the embeddings!
    #                               embed_dim=emb_dim,
    #                               # init_embedding=MTSPInitEmbedding(emb_dim),
    #                               # context_embedding=MTSPContext(emb_dim),
    #                               # dynamic_embedding=StaticEmbedding(emb_dim)
    #                               )
    policy = MultiAttentionModelPolicy(env_name=env.name,
                                       embed_dim=emb_dim,
                                       init_embedding_location=MTSPInitEmbedding(emb_dim),
                                       init_embedding_charging=ChargingInitEmbedding(emb_dim),
                                       context_embedding=MTSPContext(emb_dim),
                                       dynamic_embedding=StaticEmbedding(emb_dim)
                                       )
    model = AttentionModel(env,
                           baseline='rollout',
                           policy=policy,
                           train_data_size=10_000,  # really small size for demo
                           val_data_size=10_000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[4]).to(device)
    policy = model.policy.to(device)
    # out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
    #
    # print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    # for td, actions in zip(td_init, out['actions'].cpu()):
    #     fig = env.render(td, actions)
    #     fig.show()

    trainer = RL4COTrainer(max_epochs=1, devices="auto")
    trainer.fit(model)

    # Greedy rollouts over trained model (same states as previous plot)
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)

    # Plotting
    print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init, out['actions'].cpu()):
        fig = env.render(td, actions)
        fig.show()


if __name__ == "__main__":
    main()
