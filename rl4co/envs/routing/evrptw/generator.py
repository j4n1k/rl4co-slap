from typing import Union, Callable

import numpy as np
import torch
from einops import repeat

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator
from rl4co.utils.ops import gather_by_index

log = get_pylogger(__name__)

class EVRPGenerator(Generator):
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
        instance_dm: torch.tensor = None,
        n_customers: int = 12,
        n_charging_stations: int = 1,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        min_dist: float = 1.0,
        max_dist: float = 2.0,
        min_time: float = 0.0,
        max_time: float = 30,
        h_init: float = 3.9,
        h_max: float = 3.9,
        h_final: float = 3.9 * 0.2,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        min_num_agents: int = 4,
        max_num_agents: int = 4,
        dmat_only: bool = True,
        dist_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        tmat_class: bool = True,
        scale: bool = False,
        **kwargs,
    ):
        self.n_customers = n_customers
        self.n_charging_stations = n_charging_stations
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.min_time = min_time
        self.max_time = max_time
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents
        self.h_init = h_init
        self.h_max = h_max
        self.h_final = h_final
        self.num_loc = self.n_customers + self.n_charging_stations + (max_num_agents * 2)
        self.scale = scale
        self.dmat_only = dmat_only
        self.instance_dm = instance_dm

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
        # Sample the number of agents
        num_agents = torch.randint(
            self.min_num_agents,
            self.max_num_agents + 1,
            size=(*batch_size,),
        )
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        if isinstance(self.instance_dm, np.ndarray):
            dms = torch.tensor(self.instance_dm).float() / 1000
            dms = repeat(dms, 'n d -> b n d', b=batch_size[0], d=dms.shape[0])
            # dms = self.instance_dm
        else:
            if self.dmat_only:
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
            else:
                dms = self._get_distance_matrix(locs)
        # look-ups for different node types
        nodes = torch.arange(self.num_loc).repeat(batch_size).view(batch_size[0], self.num_loc)
        customer_nodes = nodes[..., 0:self.n_customers]
        depot_nodes = nodes[..., self.n_customers+1:]
        charging_node = nodes[..., self.n_customers]
        n_depots = depot_nodes.shape[1]
        start_depots = depot_nodes[..., 0:n_depots//2]
        end_depots = depot_nodes[..., n_depots//2:]
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        discrete_values = torch.arange(5, 35, 5)
        samples = torch.multinomial(probs, num_samples=customer_nodes.shape[-1] * batch_size[0], replacement=True).view(batch_size[0], customer_nodes.shape[-1])
        due_dates = discrete_values[samples]
        due_dates, _ = torch.sort(due_dates, dim=1, descending=False)
        durations = torch.zeros(
            *batch_size, self.num_loc, dtype=torch.float32
        )
        ## define time windows
        # 1. get distances from depot
        dist = gather_by_index(dms, depot_nodes).max()

        # 2. define upper bound for time windows to make sure the vehicle can get back to the depot in time
        upper_bound = self.max_time - dist - durations

        # 3. create random values between 0 and 1
        ts_1 = torch.rand(*batch_size, self.num_loc)
        ts_2 = torch.rand(*batch_size, self.num_loc)

        # 4. scale values to lie between their respective min_time and max_time and convert to integer values
        min_ts = (dist + (upper_bound - dist) * ts_1).int()
        max_ts = (dist + (upper_bound - dist) * ts_2).int()

        # 5. set the lower value to min, the higher to max
        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)

        # 6. reset times for depots and charging node
        min_times[..., :, depot_nodes[0]] = 0.0
        min_times[..., :, charging_node[0]] = 0.0
        min_times[..., :, customer_nodes[0]] = 0.0
        max_times[..., :, depot_nodes[0]] = self.max_time
        max_times[..., :, charging_node[0]] = self.max_time

        # 7. ensure min_times < max_times to prevent numerical errors in attention.py
        # min_times == max_times may lead to nan values in _inner_mha()
        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(
                dist.int(), min_tmp[mask] - 1
            )  # we are handling integer values, so we can simply substract 1
            min_times = min_tmp

            mask = min_times == max_times  # update mask to new min_times
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]).int(),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]).int(),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        # 8. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # Reset duration at depot to 0
        durations[:, 0] = 0.0
        # h_max = self.h_max
        # h_final = self.h_final
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
                "end_depots": end_depots,
                "due_dates": due_dates,
                "time_windows": time_windows,
                "h_max": torch.full((*batch_size, 1), self.h_max),
                "h_final": torch.full((*batch_size, 1), self.h_final),
                "h_init": torch.full((*batch_size, 1), self.h_init)
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