from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length

from .generator import EVRPGenerator
from .render import render

class EVRPEnv(RL4COEnvBase):
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

    name = "tsp"

    def __init__(
        self,
        generator: EVRPGenerator = None,
        generator_params: dict = {},
        cost_type: str = "minmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRPGenerator(**generator_params)
        self.generator = generator
        self.cost_type = cost_type
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        # Initial variables
        batch_size = td["action"].shape[0]
        is_first_action = batch_to_scalar(td["i"]) == 0
        current_node = td["action"].clone()
        first_node = current_node if is_first_action else td["first_node"]
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )

        # get distance traveled from last node to current node
        dist_mat = td["cost_matrix"]
        energy_mat = td["energy_consumption"]
        last_node_to_all = gather_by_index(dist_mat, idx=td["current_node"])
        last_node_to_all_energy = gather_by_index(energy_mat, idx=td["current_node"])
        dist_traveled = gather_by_index(last_node_to_all, idx=current_node)
        energy_consumed = gather_by_index(last_node_to_all_energy, idx=current_node)
        # Update current time
        td["current_time"] = torch.max(td["current_time"] + dist_traveled[:, None], start_times)
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
        td["battery"] += (current_node == agent_end_depot).long().view(td["battery"].shape) * (3.9 - td["battery"])

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
        proj_depletion = gather_by_index(energy_mat, idx=current_node)
        # proj_depletion = current_node_to_all * 10
        proj_battery = td["battery"] - proj_depletion
        reachable = proj_battery > td["threshold"]
        # Charging Station is always reachable
        reachable[..., td["charging_node"]] = 1
        # Check if lower battery threshold is reached
        battery_check = td["battery"] < td["threshold"]
        # time check
        dist = gather_by_index(dist_mat, idx=current_node)
        can_reach_in_time = (
                td["current_time"] + dist <= td["time_windows"][..., 1]
        )
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
            action_mask = available & reachable # & can_reach_in_time
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
        current_duration[cs_mask] += 3

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
        td["current_time"] *= (cur_agent_idx[:, None] == td["agent_idx"][:, None]).float()

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
        # battery = torch.full((*batch_size, 1), 3.9, dtype=torch.float32)
        # threshold = torch.full(
        #     (*batch_size, 1), 0.781, device=device
        # )
        battery = td["h_init"]
        threshold = td["h_final"]
        values = torch.tensor([3.9], device=device)
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
                "thresholds": thresholds,
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                )
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

    def _make_spec(self, generator: EVRPGenerator):
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