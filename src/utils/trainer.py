import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils

import wandb

from tqdm import tqdm

class Trainer:
    def __init__(
            self, 
            run_name: str,
            model_generator, model_discriminator, 
            optimizer_generator, optimizer_discriminator,
            criterion,
            train_loader, test_loader,
            lr_scheduler_generator = None, lr_scheduler_discriminator = None,
            device = None, start_epoch: int = 1, len_epoch: int = None) -> None:

        self.model_g = model_generator
        self.model_d = model_discriminator
        self.optimizer_g = optimizer_generator
        self.optimizer_d = optimizer_discriminator
        self.lr_scheduler_g = lr_scheduler_generator
        self.lr_scheduler_d = lr_scheduler_discriminator
        self.criterion = criterion

        self.start_epoch = start_epoch
        self.train_loader = train_loader
        self.test_loader = test_loader
        if len_epoch is None:
            len_epoch = len(train_loader)
        self.len_epoch = len_epoch
        self.device = device
        
        self.epoch = start_epoch

        if wandb.run is not None:
            wandb.finish()
        
        wandb.init(
            project='gan',
            name=run_name
        )

        self.step = 0

    def _train_epoch(self):
        self.model_g.train()
        self.model_d.train()
        for image, _ in tqdm(self.train_loader, total=self.len_epoch, desc=f'Epoch: {self.epoch}'):
            real_image = image.to(self.device)
            batch_size = real_image.size()[0]
            real_label = torch.ones((batch_size, 1), dtype=torch.float).to(self.device)
            fake_label = torch.zeros((batch_size, 1), dtype=torch.float).to(self.device)

            # Update discriminator first
            self.optimizer_d.zero_grad()
            logit_real = self.model_d(real_image)
            d_loss_real = self.criterion(logit_real, real_label)
            d_loss_real.backward()

            fake_image = torch.randn(size=(batch_size, 100)).to(self.device)
            fake_image = self.model_g(fake_image)
            logit_fake = self.model_d(fake_image.detach())
            d_loss_fake = self.criterion(logit_fake, fake_label)
            d_loss_fake.backward()
            d_loss = d_loss_real.item() + d_loss_fake.item()
            self.optimizer_d.step()

            # Update generator now
            self.optimizer_g.zero_grad()
            logit_fake = self.model_d(fake_image)
            g_loss = self.criterion(logit_fake, real_label)
            g_loss.backward()
            self.optimizer_g.step()

            g_loss = g_loss.item()

            wandb.log({
                'g_loss': g_loss, 'd_loss': d_loss
            }, step=self.step)
            self.step += 1

    @torch.no_grad()
    def _generate_examples(self):
        self.model_g.eval()
        fake_image = torch.randn(size=(64, 100)).to(self.device)
        fake_image = self.model_g(fake_image)

        grid = vutils.make_grid(fake_image, nrow=8, padding=2, normalize=True)
        wandb.log({'generated_images': wandb.Image(grid)}, step=self.step)
            
    def train(self, n_epoch: int):
        for epoch in range(self.start_epoch, n_epoch + 1):
            self.epoch = epoch
            self._train_epoch()
            self._generate_examples()
