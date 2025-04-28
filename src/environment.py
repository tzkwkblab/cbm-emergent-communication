# src/environment.py

import numpy as np
import torch
from gym import Env, spaces

from .utils import translate_latent_vector

class DummyEnv(Env):
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, {}

class DynamicConceptEnv(Env):
    def __init__(self, latent_predictor, label_predictor, alignment_mapping, device, entropy_coef=0.1, env_index=0):
        super().__init__()
        self.device = torch.device(device)
        self.latent_predictor = latent_predictor
        self.label_predictor = label_predictor
        self.alignment_mapping = alignment_mapping
        self.entropy_coef = entropy_coef
        self.env_index = env_index
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(latent_predictor.projected_features_dim,),
                                             dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2] * latent_predictor.num_concepts)
        self.cached_obs = None
        self.sample_label = None

    def update_batch(self, batch_features, batch_labels):
        batch_features_tensor = torch.as_tensor(batch_features, dtype=torch.float32, device=self.device)
        batch_labels_tensor = torch.as_tensor(batch_labels, dtype=torch.long, device=self.device)
        with torch.no_grad():
            projected = self.latent_predictor.feature_extractor(batch_features_tensor)
        self.cached_obs = projected[self.env_index].unsqueeze(0)
        self.sample_label = int(batch_labels_tensor[self.env_index].item())

    def update_mapping(self, new_mapping):
        self.alignment_mapping = new_mapping

    def reset(self):
        return self.cached_obs.cpu().numpy().flatten()

    def step(self, action):
        token_vector = translate_latent_vector(action, self.alignment_mapping, self.env_index, self.hoc_matrix_np, self.latent_predictor.num_concepts)
        tokens_tensor = torch.as_tensor(token_vector, dtype=torch.float, device=self.device).unsqueeze(0)

        self.label_predictor.eval()
        with torch.no_grad():
            preds = self.label_predictor(tokens_tensor)
        predicted_class = preds.argmax(dim=1).item()
        confidence = preds.max(dim=1)[0].item()

        reward = confidence if predicted_class == self.sample_label else -confidence
        done = True
        info = {'predicted_class': predicted_class, 'confidence': confidence}
        return self.reset(), reward, done, info
