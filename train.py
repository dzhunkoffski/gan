import torch
from torch import nn

from torchvision.datasets import GTSRB
import torchvision.transforms as T
from torch.utils.data import DataLoader

transform = T.Compose([
    T.Resize(64),
    T.CenterCrop(64),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = GTSRB(root='data', split='test', download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

from src.model.dcgan.discriminator import Discriminator
from src.model.dcgan.generator import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_g = Generator(latent_dim=100)
model_d = Discriminator()
model_g = model_g.to(DEVICE)
model_d = model_d.to(DEVICE)

criterion = nn.BCELoss()
optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_g = torch.optim.Adam(model_g.parameters(), lr=0.0002, betas=(0.5, 0.999))

from src.utils.trainer import Trainer
import os

os.environ['WANDB_API_KEY'] = '3dbf6bc6ee9d845aeb9d49a5be8ef71e8c9fc466'

t = Trainer(
    run_name='demo_run',
    model_discriminator=model_d,
    model_generator=model_g,
    optimizer_generator=optimizer_g,
    optimizer_discriminator=optimizer_d,
    criterion=criterion,
    train_loader=train_loader,
    test_loader=train_loader,
    device=DEVICE
)
t.train(n_epoch=10)
