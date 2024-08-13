from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedDiscreteTensorSpec

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import SLAPGenerator

log = get_pylogger(__name__)


class SLAPEnv(RL4COEnvBase):
    """
    observation contains product frequency, next product to assign, distance matrix

    """
    name = "slap"

    def __init__(
            self,
            generator: SLAPGenerator = None,
            generator_params: dict = {},
            check_solution=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SLAPGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # always place the next item / first item in list "to_choose"
        product_idx = 0
        product = td["to_choose"][..., product_idx]
        # cut off the just chosen product
        to_choose = td["to_choose"][..., product_idx+1:]

        # chosen = td["chosen"].clone()
        # chosen[torch.arange(batch_size).to(td.device), selected] = True
        location_idx = td["action"]
        batch_size = location_idx.shape[0]

        assignment = td["assignment"].clone()
        # TODO fix datatypes
        product = product.to(torch.int)
        location_idx = location_idx.to(torch.int)
        assignment[torch.arange(batch_size).to(td.device), product] = location_idx
        n_products = td["freq"].shape[-2]
        # done if we have placed all products
        done = td["i"] == n_products - 1
        # mask = assignment == torch.tensor(-1.)
        # set unavailable locations to False
        mask = td["action_mask"].clone()
        for batch in range(batch_size):
            mask[batch][location_idx[batch]] = False
        # dist_mat = self._get_distance_matrix(td["locs"])
        # depot_idx = 0
        # depot_to_all_distances = dist_mat[:, depot_idx, :]
        # next_product = td["to_choose"][..., 0]
        # next_product = next_product.to(torch.int)
        # next_product_freq = td["freq"][
        #     torch.arange(td["freq"].size(0)
        #                  ).unsqueeze(0),
        #     next_product].to(torch.int)
        # n_locs = td["depot_loc_dist"].shape[1]
        # val1 = next_product_freq[0, 0, 0].item()
        # val2 = next_product_freq[0, 1, 0].item()
        # first_row = torch.full((n_locs,), val1, dtype=torch.int32)  # Shape: [100]
        # second_row = torch.full((n_locs,), val2, dtype=torch.int32)  # Shape: [100]
        #
        # final_tensor = torch.stack((first_row, second_row), dim=0)
        # # print(f"{final_tensor.shape} / {td['depot_loc_dist'].shape}")
        # ratio = td["depot_loc_dist"] / next_product_freq.unsqueeze(-1)
        # ratio[..., 0] = 0
        td.update(
            {
                # states changed by actions
                "assignment": assignment,
                "to_choose": to_choose,
                "action_mask": mask,
                "i": td["i"] + 1,  # the number of products we have chosen
                "reward": torch.zeros_like(done),
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize empty/ unassigned warehouse
        init_assignment = td["assignment"] if td is not None else None
        if batch_size is None:
            batch_size = self.batch_size if init_assignment is None else init_assignment.shape[:-2]
        device = init_assignment.device if init_assignment is not None else self.device
        self.to(device)
        if init_assignment is None:
            init_assignment = self.generate_data(batch_size=batch_size).to(device)["assignment"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        n_products = td["freq"].shape[-2]
        to_choose = torch.arange(n_products, dtype=torch.float32).unsqueeze(0).repeat(*batch_size, 1)

        # available contains the storage locations to select from. Initially all locations are set to
        # "True" and can be selected
        available = torch.ones(
            (*batch_size, td["locs"].shape[1]), dtype=torch.bool, device=device
        )
        ratio = torch.zeros(td["depot_loc_dist"].shape)
        # Depot is not available
        available[..., 0] = False
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "assignment": init_assignment,
                "to_choose": to_choose,
                "i": i,
                "ratio": ratio,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _get_reward(self, td, actions) -> TensorDict:
        assignment = td["assignment"]
        orders = td["picklist"]
        n_batches = orders.shape[0]
        total_reward = torch.full((n_batches,),0, dtype=torch.float32)
        for i in range(orders.shape[1]):
            subset = assignment[
                torch.arange(assignment.size(0)
                             ).unsqueeze(1),
                orders[:, i, :]].to(torch.int)
            locs_orders = td["locs"][torch.arange(td["locs"].size(0)).unsqueeze(1), subset]
            total_reward += -get_tour_length(locs_orders)
        return total_reward

    def _make_spec(self, generator: SLAPGenerator):
        # TODO: make spec
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.generator.n_locs,
        )

    def render(self, td, actions=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        td = td.detach().cpu()

        locs = td["locs"]
        assigned = td["locs"][actions]
        x, y = locs[:, 0], locs[:, 1]
        x_a, y_a = assigned[:, 0], assigned[:, 1]

        # Plot the visited nodes
        ax.scatter(x, y, color="tab:blue")
        ax.scatter(x_a, y_a, color='red', s=100, label='Assigned Products',
                    edgecolor='black')
        # for i, idx in enumerate(assigned):
        #     ax.annotate(
        #         assigned[i],
        #         (x_a[idx], y_a[idx]),
        #         textcoords="offset points",
        #         xytext=(0, 10),
        #         ha='center',
        #         fontsize=12,
        #         color='black'
        #     )
        plt.show()


