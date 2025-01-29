import math
import random

from typing import Callable, Union

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch import Tensor
from torch.distributions import Uniform
from torch.utils.data import WeightedRandomSampler

from rl4co.utils.ops import gather_by_index
from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SLAPGenerator(Generator):
    def __init__(
            self,
            n_products: int = 20,
            n_aisles: int = 10,
            n_locs: int = 10,
            inter_loc_dist: float = 1,
            inter_aisle_dist: float = 2.4,
            min_freq: int = 1,
            max_freq: int = 20,
            max_orders: int = 20,
            max_products_in_order: int = 5
    ):
        self.n_locs = n_locs
        self.n_products = n_products
        self.n_aisles = n_aisles
        self.n_locs = n_locs
        self.max_orders = max_orders
        self.max_products_in_order = max_products_in_order
        self.inter_loc_dist = inter_loc_dist
        self.inter_aisle_dist = inter_aisle_dist
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_sampler = torch.distributions.Uniform(
            low=min_freq, high=max_freq
        )
        # self.freq_sampler = torch.distributions.Pareto(min_freq, 2.0)
        self.order_sampler = WeightedRandomSampler

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

    def _calc_coordinates(self, batch_size):
        total_locations = self.n_aisles * self.n_locs

        # Initialize tensor to hold coordinates
        coordinates_tensor = torch.zeros((*batch_size, total_locations, 2), dtype=torch.float32)

        # Fill the tensor with coordinates
        for b in range(batch_size[0]):
            for i in range(total_locations):
                aisle = i // self.n_locs
                location = i % self.n_locs
                y = location * self.inter_loc_dist
                x = aisle * self.inter_aisle_dist
                coordinates_tensor[b, i] = torch.tensor([x, y], dtype=torch.float32)
        return coordinates_tensor

    def _sample_order(self, freq: Tensor):
        # order = self.order_sampler(freq,
        #                    self.max_products_in_order,
        #                    replacement=False)
        n_products = random.randint(1, self.max_products_in_order)
        order = [torch.multinomial(freq.squeeze(-1), n_products, replacement=False) for _ in range(self.max_orders)]
        return order

    def _create_picklist(self, freq):
        # n_orders = random.randint(5, self.max_orders)
        # picklist = torch.zeros((*batch_size, n_orders))
        # for b in range(len(batch_size)):
        #     for i in range(n_orders):
        #         picklist[b, i] = self._sample_order(freq)
        # n_products = random.randint(1, self.max_products_in_order)
        # picklist = torch.multinomial(freq.squeeze(-1), n_products, replacement=False)
        order_batches = []
        n_batches = freq.squeeze(-1)
        for _ in n_batches:
            orders = []
            for _ in range(self.max_orders):
                # Random number of items in this order
                # num_items = np.random.randint(1, self.max_products_in_order + 1)
                # Generate random SKUs for this order
                skus = np.random.randint(0, self.n_products, size=self.max_products_in_order).tolist()
                orders.append(skus)
            padded_orders = [order + [-1] * (self.max_products_in_order - len(order)) for order in orders]
            order_batches.append(padded_orders)
        picklist = torch.tensor(order_batches)
        return picklist

    def _get_order_freq(self, picklist):
        num_batches = picklist.shape[0]

        # List to store counts for each batch
        batch_counts = []

        # Process each batch
        for i in range(num_batches):
            batch = picklist[i]

            # Flatten the current batch
            flattened = batch.view(-1)

            # Count occurrences of each position
            counts = torch.bincount(flattened)

            # Store counts for the current batch
            batch_counts.append(counts)

        # Convert list to a tensor for easier handling
        order_freq = torch.stack(batch_counts)
        return order_freq

    def _generate(self, batch_size) -> TensorDict:
        item_probabilities = self.freq_sampler.sample((*batch_size, self.n_products, 1))
        locs = self._calc_coordinates(batch_size)
        dist_mat = self._get_distance_matrix(locs)
        depot_idx = 0
        depot_to_all_distances = dist_mat[:, depot_idx, :]

        # assignment = torch.full((*batch_size, self.n_locs * self.n_aisles), -1, dtype=torch.float32)
        assignment = torch.full((*batch_size, self.n_products), -1, dtype=torch.int)
        picklist = self._create_picklist(item_probabilities)
        # freq = self._get_order_freq(picklist)
        td = TensorDict({"freq": item_probabilities,
                         "locs": locs,
                         "dist_mat": dist_mat,
                         "assignment": assignment,
                         "picklist": picklist,
                         "depot_loc_dist": depot_to_all_distances},
                        batch_size=batch_size)
        return td


