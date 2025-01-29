import torch
import torch.nn as nn
from rl4co.envs import SLAPEnv, RL4COEnvBase, OrderBatchingEnv
from rl4co.envs.warehousing.slap.generator import SLAPGenerator
from rl4co.models import AttentionModelPolicy, REINFORCE, AttentionModel
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils.ops import gather_by_index

env = OrderBatchingEnv(generator_params={'n_aisles': 10,
                                'n_locs': 10})

emb_dim = 128
policy = AttentionModelPolicy(env_name="obp", # this is actually not needed since we are initializing the embeddings!
                              embed_dim=emb_dim
)

model = AttentionModel(env,
                       policy=policy,
                       baseline='rollout',
                       train_data_size=20,
                       val_data_size=20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[3]).to(device)
policy = model.policy.to(device)
out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)