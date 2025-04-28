# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from .environment import DummyEnv

class CustomFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2048, output_dim=256):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class LatentConceptPredictor:
    def __init__(self, num_features=2048, projected_features=256, num_concepts=30,
                 num_envs=32, n_steps=32, device="cuda"):
        self.device = torch.device(device)
        self.num_concepts = num_concepts
        self.projected_features_dim = projected_features
        self.num_envs = num_envs
        self.n_steps = n_steps

        self.feature_extractor = CustomFeatureExtractor(num_features, projected_features).to(self.device)

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(projected_features,), dtype=np.float32)
        act_space = spaces.MultiDiscrete([2] * num_concepts)

        dummy_env = DummyVecEnv([lambda: DummyEnv(obs_space, act_space) for _ in range(num_envs)])
        self.ppo_model = PPO(
            ActorCriticPolicy,
            dummy_env,
            policy_kwargs=dict(net_arch=dict(pi=[512,256], vf=[512,256]), activation_fn=nn.ReLU),
            n_steps=n_steps,
            batch_size=16,
            ent_coef=0.01,
            learning_rate=3e-4,
            device=self.device,
            verbose=0
        )

    def get_projected_features(self, raw_features):
        raw_tensor = torch.as_tensor(raw_features, dtype=torch.float32, device=self.device)
        return self.feature_extractor(raw_tensor)

    def get_action(self, raw_features):
        projected = self.get_projected_features(raw_features).detach().cpu().numpy()
        actions, _ = self.ppo_model.predict(projected, deterministic=True)
        return actions

class LabelPredictor(nn.Module):
    def __init__(self, num_classes, num_concepts=30):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, x):
        return self.linear(x)