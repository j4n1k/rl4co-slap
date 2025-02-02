import torch
import torch.nn as nn
from rl4co.envs import SLAPEnv
from rl4co.envs.warehousing.slap.generator import SLAPGenerator
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.models import AttentionModel
from rl4co.utils.trainer import RL4COTrainer


class SLAPInitEmbedding(nn.Module):
    """Initial embedding
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SLAPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, dist_to_depot
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)  # locs

    def forward(self, td):
        dist_mat = self._get_distance_matrix(td["locs"])
        depot_idx = 0
        depot_to_all_distances = dist_mat[:,depot_idx, :]
        node_embeddings = self.init_embed(
            torch.cat((td["locs"], depot_to_all_distances[..., None]), -1)
        )
        print(node_embeddings.shape)
        return node_embeddings

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


class SLAPContext(EnvContext):
    """Context embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Project the following to the embedding space:
        - current cell embedding
    """

    def __init__(self, embed_dim):
        super(SLAPContext, self).__init__(embed_dim)

    def forward(self, embeddings, td):
        """Context cannot be defined by a single node embedding for DPP, hence 0.
        We modify the dynamic embedding instead to capture placed items
        """
        return embeddings.new_zeros(embeddings.size(0), self.embed_dim)


class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


if __name__ == "__main__":
    env = SLAPEnv(generator_params={'n_aisles': 10,
                                    'n_locs': 10})

    emb_dim = 128
    policy = AttentionModelPolicy(env_name=env.name,
                                  embed_dim=emb_dim,
                                  init_embedding=SLAPInitEmbedding(emb_dim),
                                  context_embedding=SLAPContext(emb_dim),
                                  dynamic_embedding=StaticEmbedding(emb_dim)
                                  )

    model = AttentionModel(env,
                           policy=policy,
                           baseline='rollout',
                           train_data_size=100_000,
                           val_data_size=10_000)

    trainer = RL4COTrainer(max_epochs=1, devices="auto")
    trainer.fit(model)
