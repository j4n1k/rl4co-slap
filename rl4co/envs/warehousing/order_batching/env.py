from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedDiscreteTensorSpec

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import OrderBatchingGenerator

log = get_pylogger(__name__)


class OrderBatchingEnv(RL4COEnvBase):
    """Order Batching Problem (OBP) environment with batch tracking.

    The agent selects orders and assigns them to batches while considering constraints such as batch volume and the warehouse layout.
    The goal is to minimize the total picking time and the number of batches.

    Observations:
        - Warehouse layout (item positions).
        - Orders with their respective items and volume.
        - Current batch assignment and remaining capacity of each batch.

    Constraints:
        - Batch cannot exceed the maximum capacity.
        - Orders must be fully assigned to a batch.
        - The picking sequence can affect the reward.

    Finish Condition:
        - All orders are assigned to batches.

    Reward:
        - The negative total picking time and the number of batches (minimization).

    Args:
        generator: OrderBatchingGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "order_batching"

    def __init__(self, generator: OrderBatchingGenerator = None, generator_params: dict = {}, **kwargs):
        super().__init__(**kwargs)
        if generator is None:
            generator = OrderBatchingGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_order = td["action"][:, None]  # Action: select an order for batching
        selected_items = gather_by_index(td["orders"], current_order, squeeze=False)

        # Update the current batch capacity
        updated_batch_cap = td["batch_capacity"] - selected_items.sum(-1, keepdim=True)

        # Check if the batch is full and create a new batch if needed
        batch_full = updated_batch_cap < 0
        new_batch_created = batch_full.float() * self.generator.max_batch_capacity

        # Adjust the current batch capacity: set to the max capacity if a new batch is created
        batch_capacity = updated_batch_cap + new_batch_created

        # Track batch assignment: assign the order to the current batch
        batch_idx = td["batch_idx"].clone()
        batch_idx[batch_full] += 1  # Increment batch index if new batch created

        assigned_orders = td["assigned_orders"].scatter(-1, current_order, batch_idx)

        done = (td["assigned_orders"].sum(-1) == td["orders"].size(-1))  # Done when all orders are assigned
        reward = torch.zeros_like(done)

        td.update({
            "current_order": current_order,
            "batch_capacity": batch_capacity,
            "assigned_orders": assigned_orders,
            "batch_idx": batch_idx,
            "reward": reward,
            "done": done,
        })
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        n_orders = td["orders"].shape[1]
        # all orders can be selected
        action_mask = torch.ones(
            (*batch_size, n_orders), dtype=torch.bool, device=device
        )
        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "orders": td["orders"],
                "orders_to_assign": torch.arange(n_orders,
                                                 dtype=torch.float32).unsqueeze(0).repeat(*batch_size, 1),
                "current_order": torch.zeros(*batch_size, 1, dtype=torch.long, device=device),
                "batch_capacity": torch.full((*batch_size, 1), self.generator.max_batch_capacity, device=device),
                "assigned_orders": torch.zeros((*batch_size, td["orders"].shape[1]), dtype=torch.int64, device=device),
                "batch_idx": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device),
                "action_mask": action_mask
            },
            batch_size=batch_size,
        )
        # td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # Mask orders that are already assigned or exceed the batch capacity
        exceeds_cap = td["orders"] + td["batch_capacity"] > td["batch_capacity"]
        mask_orders = td["assigned_orders"] > 0 | exceeds_cap
        return ~mask_orders

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Reward is based on minimizing picking time and batch count
        picking_time = self.compute_picking_time(td, actions)
        batch_count = td["batch_idx"].max(-1)[0].float() + 1  # Number of batches used
        return -(picking_time + batch_count)

    def compute_picking_time(self, td: TensorDict, actions: TensorDict) -> torch.Tensor:
        # Compute the total picking time based on warehouse layout and item positions
        positions = gather_by_index(td["positions"], actions)
        return self.generator.compute_picking_time(positions)

    def _make_spec(self, generator: OrderBatchingGenerator):
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.generator.n_orders,
        )
    #     self.observation_spec = CompositeSpec(
    #         orders=BoundedTensorSpec(
    #             low=0,
    #             high=generator.max_order_volume,
    #             shape=(generator.num_orders, generator.num_items_per_order),
    #             dtype=torch.float32,
    #         ),
    #         current_order=UnboundedDiscreteTensorSpec(
    #             shape=(1),
    #             dtype=torch.int64,
    #         ),
    #         batch_capacity=BoundedTensorSpec(
    #             low=0,
    #             high=generator.max_batch_capacity,
    #             shape=(1,),
    #             dtype=torch.float32,
    #         ),
    #         assigned_orders=UnboundedDiscreteTensorSpec(
    #             shape=(generator.num_orders,),
    #             dtype=torch.int64,  # Tracks the batch index of each order
    #         ),
    #         batch_idx=UnboundedDiscreteTensorSpec(
    #             shape=(1,),
    #             dtype=torch.int64,  # Tracks the current batch index
    #         ),
    #         action_mask=UnboundedDiscreteTensorSpec(
    #             shape=(generator.num_orders,),
    #             dtype=torch.bool,
    #         ),
    #         shape=(),
    #     )
    #     self.action_spec = BoundedTensorSpec(
    #         shape=(1,),
    #         dtype=torch.int64,
    #         low=0,
    #         high=generator.num_orders,
    #     )
    #     self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
    #     self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        pass

