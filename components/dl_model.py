import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class ActorCriticCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 84, 84)
            cnn_out_size = self.cnn(dummy).view(1, -1).size(1)
        # MLP heads
        self.actor = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # Preprocessing transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])

    def preprocess(self, obs_batch):
        # obs_batch: numpy array or torch tensor, shape (N, H, W, C) or (N, C, H, W)
        if isinstance(obs_batch, torch.Tensor):
            obs_batch = obs_batch.cpu().numpy()
        imgs = []
        for obs in obs_batch:
            if obs.shape[-1] == 3:
                # (H, W, C)
                img = self.transform(obs.astype('uint8'))
            else:
                # (C, H, W)
                img = self.transform(obs.transpose(1, 2, 0).astype('uint8'))
            imgs.append(img / 255.0)  # Normalize to [0, 1]
        return imgs

    def forward(self, obs):
        # obs: batch of images (N, H, W, C) or (N, C, H, W)
        imgs = self.preprocess(obs)
        x = torch.stack(imgs).to(next(self.parameters()).device)
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def act(self, logits):
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample().unsqueeze(-1)
        return actions, dist, probs
