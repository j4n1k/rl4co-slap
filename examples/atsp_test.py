from rl4co.envs.routing.atsp.env import ATSPEnv
from rl4co.models import MatNetPolicy, MatNet
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
from rl4co.models.zoo import EAS, EASLay, EASEmb, ActiveSearch
from rl4co.utils.decoding import get_log_likelihood
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
import networkx as nx

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
        if actions.dim() > 1:
            actions = actions[0]
    num_agents = td["num_agents"]
    # locs = td["locs"]
    cmap = discrete_cmap(num_agents, "rainbow")

    fig, ax = plt.subplots()
    rows, cols = 4, 3  # 4 rows and 3 columns
    rectangular_spread = [(x, y) for x in np.linspace(-1.5, 1.5, cols) for y in np.linspace(-1.5, 1.5, rows)]
    top_point = [(0, 3)]  # Point 12 on top

    left_points = [(-3, 0.5), (-3, 0), (-3, -0.5), (-3, -1)]  # Points 13-16 to the left

    right_points = [(3, 0.5), (3, 0), (3, -0.5), (3, -1)]  # Points 17-20 to the right
    # Updating all points
    all_points = rectangular_spread + top_point + left_points + right_points

    # Plotting the updated points
    #plt.figure(figsize=(8, 8))
    x, y = zip(*all_points)
    plt.scatter(x, y, color='blue')

    # Annotating the points with their indices
    for i, (x_coord, y_coord) in enumerate(all_points):
        plt.text(x_coord, y_coord, str(i), fontsize=12, ha='right', color='red')

    # Setting axis limits and labels for clarity
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.title("21 Points with Rectangular Spread in the Middle")
    plt.grid(True)

    # Add depot action = 0 to before first action and after last action
    # actions = torch.cat(
    #     [
    #         torch.zeros(1, dtype=torch.int64),
    #         actions,
    #         torch.zeros(1, dtype=torch.int64),
    #     ]
    # )

    # Make list of colors from matplotlib
    for i, loc in enumerate(all_points):
        if i in td["end_depots"].numpy():
            # depot
            marker = "s"
            color = "g"
            label = "End Depot"
            markersize = 10
        elif i in td["start_depots"].numpy():
            # depot
            marker = "s"
            color = "g"
            label = "Start Depot"
            markersize = 10
        elif i == 12:
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
    actions_result = []
    start_depots = td["start_depots"]
    end_depots = td["end_depots"]
    actions_result.append(start_depots[0])
    start_depots = start_depots[1:]
    for action in actions:
        actions_result.append(action)
        if action in end_depots and start_depots.size()[0] > 0:
            actions_result.append(start_depots[0])
            start_depots = start_depots[1:]
    if start_depots.size()[0] == 0 and end_depots[-1] not in actions_result:
        actions_result.append(end_depots[-1])

    # Plot the actions in order
    agent_idx = 1
    agent_travel_time = 0
    max_travel = 0
    n_charges = {i+1: 0 for i in range(num_agents)}
    battery_levels = {i+1: [100] for i in range(num_agents)}

    for i in range(len(actions_result)):
        if actions_result[i] in end_depots:
            agent_idx += 1
            idx = list(end_depots.numpy()).index(actions_result[i].item())
            ax.annotate(str(agent_travel_time), right_points[idx])
            if agent_travel_time > max_travel:
                max_travel = agent_travel_time
            agent_travel_time = 0
            continue

        color = cmap(num_agents - agent_idx)

        from_node = actions_result[i]
        to_node = actions_result[i + 1]
        # to_node = (
        #     actions_result[i + 1] if i < len(actions_result) - 1 else actions_result[0]
        # )  # last goes back to depot
        from_loc = all_points[from_node]
        to_loc = all_points[to_node]
        travel_time = td["cost_matrix"][from_node, to_node]
        if agent_idx <= num_agents:
            battery_levels[agent_idx].append(battery_levels[agent_idx][-1] - 2 * travel_time)
        if to_node == 12:
            travel_time += 300
            n_charges[agent_idx] += 1
            battery_levels[agent_idx].append(torch.tensor([100]))
        cur_battery = str(round(battery_levels[agent_idx][-1].item()))
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
        num_loc: int = 21,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        min_dist: float = 1.0,
        max_dist: float = 2.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        min_num_agents: int = 4,
        max_num_agents: int = 4,
        dist_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        tmat_class: bool = True,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )
        self.tmat_class = tmat_class

        # Distance distribution
        if kwargs.get("dist_sampler", None) is not None:
            self.dist_sampler = kwargs["dist_sampler"]
        else:
            self.dist_sampler = get_sampler("dist", dist_distribution, 0.0, 1.0, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        # customer_nodes = locs[..., 0:11, :]
        # depot_nodes = locs[..., 13:, :]
        # charging_nodes = locs[..., 12, :]


        # Sample the number of agents
        num_agents = torch.randint(
            self.min_num_agents,
            self.max_num_agents + 1,
            size=(*batch_size,),
        )

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms = (
                self.dist_sampler.sample((batch_size + [self.num_loc, self.num_loc]))
                * (self.max_dist - self.min_dist)
                + self.min_dist
        )
        dms[..., torch.arange(self.num_loc), torch.arange(self.num_loc)] = 0
        log.info("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
        if self.tmat_class:
            while True:
                old_dms = dms.clone()
                dms, _ = (
                        dms[..., :, None, :] + dms[..., None, :, :].transpose(-2, -1)
                ).min(dim=-1)
                if (dms == old_dms).all():
                    break

        nodes = torch.arange(self.num_loc).repeat(batch_size).view(batch_size[0], self.num_loc)
        customer_nodes = nodes[..., 0:12]
        depot_nodes = nodes[..., 13:]
        charging_node = nodes[..., 12]
        n_depots = depot_nodes.shape[1]
        start_depots = depot_nodes[..., 0:n_depots//2]
        end_depots = depot_nodes[..., n_depots//2:]
        # dms[..., 12:] += 5
        dms[..., :12] += 5
        dms[..., 12, 12] = 0
        return TensorDict(
            {
                "locs": locs,
                "num_agents": num_agents,
                "cost_matrix": dms,
                "travel_time": dms,
                "energy_consumption": dms * 0.25,
                "customer_nodes": customer_nodes,
                "depot_nodes": depot_nodes,
                "charging_node": charging_node,
                "start_depots": start_depots,
                "end_depots": end_depots
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
        # Initial variables
        is_first_action = batch_to_scalar(td["i"]) == 0
        current_node = td["action"].clone()
        first_node = current_node if is_first_action else td["first_node"]

        # get distance traveled from last node to current node
        dist_mat = td["cost_matrix"]
        energy_mat = td["energy_consumption"]
        last_node_to_all = gather_by_index(dist_mat, idx=td["current_node"])
        last_node_to_all_energy = gather_by_index(energy_mat, idx=td["current_node"])
        dist_traveled = gather_by_index(last_node_to_all, idx=current_node)
        energy_consumed = gather_by_index(last_node_to_all_energy, idx=current_node)
        # Update battery levels from travel
        target_battery_level = gather_by_index(td["thresholds"], td["charging_duration"])
        # td["battery"] -= 10 * dist_traveled.view(td["battery"].shape)
        td["battery"] -= energy_consumed.view(td["battery"].shape)
        # Update battery levels from recharge if current node is charging node
        cs_mask = current_node == td["charging_node"]
        td["battery"][cs_mask] = target_battery_level[cs_mask]

        # If current_node is end depot, then increment agent_idx
        agent_end_depot = gather_by_index(td["end_depots"], td["agent_idx"])
        cur_agent_idx = td["agent_idx"] + (current_node == agent_end_depot).long()
        max_agent_mask = cur_agent_idx >= td["num_agents"]
        # shave agent index if bigger than max agents (leads to indexing issues)
        cur_agent_idx[max_agent_mask] = batch_to_scalar(td["num_agents"]) - 1
        # If agent_idx is increased, vehicle is refueled (we start with a new vehicle)
        td["battery"] += (current_node == agent_end_depot).long().view(td["battery"].shape) * (3900 - td["battery"])

        # Check which target thresholds can be used (Battery can not be decharged)
        # only needed in multi discrete scenario
        comp = td["thresholds"] >= td["battery"][:, None, :]
        comp = comp.squeeze()

        # Set not visited to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node[..., None].expand_as(td["available"]), 0
        )

        # agent's end depots can always be selected as an action unless:
        # - current_node is agent end depot
        # - agent_idx is greater than num_agents -1
        # -> asserts that all nodes are visited if there are nodes left but no agents
        available[torch.arange(available.size(0)), agent_end_depot] = torch.logical_and(
            current_node != agent_end_depot, td["agent_idx"] < td["num_agents"] - 1
        )

        # Check the nodes that can be reached with current battery level
        current_node_to_all = gather_by_index(dist_mat, idx=current_node)
        proj_depletion = current_node_to_all * 10
        proj_battery = td["battery"] - proj_depletion
        reachable = proj_battery > td["threshold"]
        # Charging Station is always reachable
        reachable[..., td["charging_node"]] = 1
        # Check if lower battery threshold is reached
        battery_check = td["battery"] < td["threshold"]

        # We are done if there are no unvisited customer nodes
        done = torch.count_nonzero(gather_by_index(available, td["customer_nodes"]), dim=-1) == 0
        # If done is True, then we make the depot available again, so that it will be selected as the next node with prob 1
        available[torch.arange(available.size(0)), agent_end_depot] = torch.logical_or(done, available[
            torch.arange(available.size(0)), agent_end_depot])
        # action_mask selection
        if battery_check.any():
            # battery below threshold, needs to be recharged
            charging_mask = available.clone()
            idx = torch.squeeze(battery_check).nonzero().squeeze(0)
            charging_mask[idx] = td["charging_mask"][idx]
            action_mask = charging_mask
        else:
            # mask out customer nodes that are not reachable
            action_mask = available & reachable
            # make charging station visitable
            action_mask[..., td["charging_node"]] = 1
            if done.any():
                # Charging Station is unavailable if done
                idx = torch.squeeze(done).nonzero().squeeze(0)
                action_mask[idx, td["charging_node"][idx]] = 0
                action_mask[idx, agent_end_depot[idx]] = 1

            # make sure that charging location can not be visited if just recharged
            if cs_mask.any():
                idx = torch.squeeze(cs_mask).nonzero().squeeze(0)
                action_mask[idx, td["charging_node"][idx]] = 0
        for i in range(action_mask.shape[0]):
            if torch.all(action_mask[i] == False):
                print("ERROR")
        # Update the current subtour length
        current_length = td["current_length"] + dist_traveled
        # Update subtour service time
        current_duration = td["current_duration"] + dist_traveled
        # Update charging duration if charging action
        # current_duration[cs_mask] += 5

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

        # If current agent is different from previous agent we update the current node to the start depot of the next agent
        # TODO Check if this messes with the distance updates
        agent_change_mask = cur_agent_idx > td["agent_idx"]
        current_node[agent_change_mask] = td["start_depots"][agent_change_mask, cur_agent_idx[agent_change_mask]]

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
        #current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        current_node = td["start_depots"][..., 0]
        available = torch.ones(
            (*batch_size, self.generator.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        # available[..., 0] = 0  # Depot is not available as first node
        # start and end depots can not be selected as actions -> will be managed during step
        charging_mask = ~available
        charging_mask[..., td["charging_node"]] = 1
        available[..., td["start_depots"]] = 0
        available[..., td["end_depots"]] = 0
        # available[..., td["charging_node"]] = 0  # charging station is not available as first node

        # Battery constraints
        battery = torch.full((*batch_size, 1), 3121, dtype=torch.float32)
        threshold = torch.full(
            (*batch_size, 1), 781, device=device
        )
        values = torch.tensor([3121.0], device=device)
        n_th = values.shape[0]
        charging_duration_mask = torch.ones((*batch_size, n_th), dtype=torch.bool, device=device)
        # thresholds = torch.full((batch_size[0], 5, 1), 70.0)
        thresholds = torch.arange(
            values.min().item(), values.max().item() + 1, step=10, device=device).repeat(batch_size[0], 1).view(batch_size[0], n_th, 1)
        charging_duration = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": td["locs"],  # depot is first node
                "num_agents": td["num_agents"],
                "travel_time": td["travel_time"],
                "energy_consumption": td["energy_consumption"],
                "customer_nodes": td["customer_nodes"],
                "depot_nodes": td["depot_nodes"],
                "charging_node": td["charging_node"],
                "start_depots": td["start_depots"],
                "end_depots": td["end_depots"],
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
                "cost_matrix": td["cost_matrix"],
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
            4  # remaining_agents, current_length, max_subtour_length, distance_from_depot, battery level, dist_to_cs
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
                # self._distance_from_depot(td),
                td["battery"].view(td["current_length"].shape),
                # self._distance_from_cs(td)
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


if torch.cuda.is_available():
    accelerator = "gpu"
    batch_size = 256
    train_data_size = 2_000
    embed_dim = 128
    num_encoder_layers = 4
else:
    accelerator = "cpu"
    batch_size = 16
    train_data_size = 1_000
    embed_dim = 64
    num_encoder_layers = 2

env = MTSPEnv()
# env = ATSPEnv()
policy = MatNetPolicy()

model = MatNet(env=env,
               policy=policy,
               baseline="shared",
               batch_size=batch_size,
               train_data_size=train_data_size,
               val_data_size=1_000)

td_init = env.reset(batch_size=[2])
# out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
# for td, actions in zip(td_init, out['actions'].cpu()):
#     fig = env.render(td, actions)
#     fig.show()
# print(out["reward"])
# trainer = RL4COTrainer(max_epochs=1, devices="auto")
# trainer.fit(model)
# out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
# print(out["reward"])
#
# for td, actions in zip(td_init, out['actions'].cpu()):
#     fig = env.render(td, actions)
#     fig.show()

# eas_model = EASEmb(env, policy, env.dataset(batch_size=[4]), batch_size=4, max_iters=20, save_path="eas_sols.pt")
#
# eas_model.setup()
#
# trainer = RL4COTrainer(
#     max_epochs=1,
#     gradient_clip_val=None,
# )
#
# trainer.fit(eas_model)

actions = torch.load("eas_sols.pt")["solutions"][0].cpu()
actions = actions[:torch.count_nonzero(actions, dim=-1)] # remove trailing zeros
#state = td_dataset.cpu()[0]
print(actions)

fig = env.render(td_init, actions)
fig.show()