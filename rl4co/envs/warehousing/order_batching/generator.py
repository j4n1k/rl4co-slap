import math
import random

from typing import Callable, Union

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch import Tensor
from torch.distributions import Uniform, Distribution
from torch.utils.data import WeightedRandomSampler

from rl4co.utils.ops import gather_by_index
from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ConstantDistribution(Distribution):
    def __init__(self):
        super().__init__(batch_shape=torch.Size([]), event_shape=torch.Size([]))

    def sample(self, sample_shape=torch.Size()):
        # Always returns 1
        return torch.ones(sample_shape)

    def log_prob(self, value):
        # Log probability is undefined for a constant distribution,
        # but you can define it as 0 for value 1 and -inf otherwise
        return torch.where(value == 1, torch.zeros_like(value), torch.full_like(value, float('-inf')))


class OrderBatchingGenerator(Generator):
    """Data generator for RL-based Order Batching problem in a warehouse.

    Args:
        n_aisles: number of aisles in the warehouse.
        n_rows: number of rows in the warehouse.
        n_zones: number of zones in the warehouse.
        n_orders: number of orders to generate.
        n_warehouse_items: total number of items in the warehouse.
        dist: distribution for common items across orders.

    Returns:
        A TensorDict with keys:
            items [batch_size, num_items, 3]: locations of items in terms of aisles, rows, and zones.
            orders [batch_size, num_orders, num_items]: binary tensor indicating items included in each order.
            volumes [batch_size, num_items]: volumes of the items in the warehouse.
            params: meta information like max_container_volume, limits on rows/aisles, etc.
    """

    def __init__(
            self,
            n_aisles: int = 100,
            n_rows: int = 100,
            n_zones: int = 5,
            n_orders: int = 50,
            n_warehouse_items: int = 300,
            dist: dict = None,
            max_container_volume: float = 1000.0,
            max_orders_per_batch: int = 50,
            max_items_in_order: int = 5,
            min_freq: int = 1,
            max_freq: int = 20,
            inter_loc_dist: float = 1,
            inter_aisle_dist: float = 2.4,
            **kwargs,
    ):
        self.max_batch_capacity = 50
        self.n_aisles = n_aisles
        self.n_rows = n_rows
        self.n_zones = n_zones
        self.n_orders = n_orders
        self.n_warehouse_items = n_warehouse_items
        self.max_container_volume = max_container_volume
        self.max_orders_per_batch = max_orders_per_batch
        self.max_items_in_order = max_items_in_order
        self.inter_loc_dist = inter_loc_dist
        self.inter_aisle_dist = inter_aisle_dist

        # Distribution for common items across orders (adjustable)
        self.dist = dist if dist is not None else {2: 100, 3: 50, 4: 20, 5: 5, 6: 2}
        self.com_items_distribution = [key for key, val in self.dist.items() for _ in range(val)]
        self.freq_sampler = torch.distributions.Uniform(
            low=min_freq, high=max_freq
        )
        self.volume_sampler = ConstantDistribution()

    def _create_picklist(self, freq):
        order_batches = []
        n_batches = freq.squeeze(-1)
        for _ in n_batches:
            orders = []
            for _ in range(self.n_orders):
                # Random number of items in this order
                # num_items = np.random.randint(1, self.max_products_in_order + 1)
                # Generate random SKUs for this order
                skus = np.random.randint(0, self.n_warehouse_items, size=self.max_items_in_order).tolist()
                orders.append(skus)
            padded_orders = [order + [-1] * (self.max_items_in_order - len(order)) for order in orders]
            order_batches.append(padded_orders)
        picklist = torch.tensor(order_batches)
        return picklist  # Additional warehouse parameters

    def _calc_coordinates(self, batch_size):
        total_locations = self.n_aisles * self.n_rows

        # Initialize tensor to hold coordinates
        coordinates_tensor = torch.zeros((*batch_size, total_locations, 2), dtype=torch.float32)

        # Fill the tensor with coordinates
        for b in range(batch_size[0]):
            for i in range(total_locations):
                aisle = i // self.n_rows
                location = i % self.n_rows
                y = location * self.inter_loc_dist
                x = aisle * self.inter_aisle_dist
                coordinates_tensor[b, i] = torch.tensor([x, y], dtype=torch.float32)
        return coordinates_tensor

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

    def _generate(self, batch_size) -> TensorDict:
        # batch_size = batch_size[0]
        item_probabilities = self.freq_sampler.sample((*batch_size, self.n_warehouse_items, 1))
        volumes = self.volume_sampler.sample((*batch_size, self.n_warehouse_items, 1))

        # def nbr_com_items_distribution() -> int:
        #     return random.choice(self.com_items_distribution)
        #
        # def article_volume() -> float:
        #     return max(1, round(random.gammavariate(2.0, 20.0)))
        #
        # # Generate item volumes
        # volumes = torch.tensor([article_volume() for _ in range(self.n_warehouse_items)])
        # Generate warehouse items' locations: aisles, rows, zones
        locs = self._calc_coordinates(batch_size)
        dist_mat = self._get_distance_matrix(locs)

        items = torch.zeros((*batch_size, self.n_warehouse_items, 3))
        for i in range(self.n_warehouse_items):
            items[:, i, 0] = random.choice(range(self.n_aisles))  # Aisle
            items[:, i, 1] = random.choice(range(self.n_rows))  # Row

        # Generate orders (binary matrix representing orders with items)
        orders = self._create_picklist(item_probabilities)

        return TensorDict(
            {
                "freq": item_probabilities,
                "locs": locs,
                "dist_mat": dist_mat,
                "items": items,  # Location info of each item [aisle, row, (zone)]
                "orders": orders,  # Binary matrix of items assigned to orders
                "volumes": volumes,  # Volume of each item
            },
            batch_size=batch_size,
        )
