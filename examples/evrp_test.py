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

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.models import AttentionModelPolicy
from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.utils import RL4COTrainer
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length
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
        # actionspace multi dimensional -> [[Locations], [charging_duration]]
        dist_mat = td["dist_mat"]
        # Initial variables
        is_first_action = batch_to_scalar(td["i"]) == 0
        last_node_to_all = gather_by_index(dist_mat, idx=td["current_node"])
        # distance traveled from last node to current node
        dist_traveled = gather_by_index(last_node_to_all, idx=td["action"])
        td["battery"] -= 10 * dist_traveled.view(td["battery"].shape)
        cs_mask = td["action"] == 1
        # if traveled to charging station, recharge to 100
        td["battery"][cs_mask] = 100
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
                print("ERROR")
        # Update the current length
        current_length = td["current_length"] + get_distance(cur_loc, prev_loc)

        # Update service time
        current_duration = td["current_duration"] + get_distance(cur_loc, prev_loc)
        current_duration[cs_mask] += 300

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
        dist_mat = self._get_distance_matrix(td["locs"])
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
                "dist_mat": dist_mat,
                "battery": battery,
                "threshold": threshold,
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


def main():
    env = MTSPEnv()
    emb_dim = 128
    policy = AttentionModelPolicy(env_name=env.name,
                                  # this is actually not needed since we are initializing the embeddings!
                                  embed_dim=emb_dim,
                                  context_embedding=MTSPContext(emb_dim),
                                  )
    model = AttentionModel(env,
                           baseline='rollout',
                           train_data_size=100_000,  # really small size for demo
                           val_data_size=10_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[4]).to(device)
    # policy = model.policy.to(device)
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)

    # Plotting
    print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init, out['actions'].cpu()):
        fig = env.render(td, actions)
        fig.show()

    trainer = RL4COTrainer(max_epochs=1, devices="auto")
    trainer.fit(model)

    # Greedy rollouts over trained model (same states as previous plot)
    policy = model.policy.to(device)
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)

    # Plotting
    print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init, out['actions'].cpu()):
        fig = env.render(td, actions)
        fig.show()


if __name__ == "__main__":
    main()
