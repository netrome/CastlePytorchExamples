import torch
import torch.nn as nn

from utils import Flatten, Unflatten, Visualizer, get_dataloader

import settings

# Discriminator network  -----------------------------------------
D = nn.Sequential(
        nn.Conv2d(1, 4, 4, stride=2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(4, 8, 4, stride=2),
        nn.LeakyReLU(0.2),
        Flatten(),
        nn.Linear(200, 10),
        nn.LeakyReLU(0.2),
        nn.Linear(10, 1),
        nn.Sigmoid(),
        )

# Generator network  -----------------------------------------
G = nn.Sequential(
        nn.Linear(10, 200),
        nn.LeakyReLU(0.2),
        Unflatten(),
        nn.ConvTranspose2d(8, 4, 5, stride=2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(4, 1, 4, stride=2),
        nn.Sigmoid(),
        )

opt_D = torch.optim.Adam(D.parameters())
opt_G = torch.optim.Adam(G.parameters())

latent_point = torch.FloatTensor(settings.batch_size, 10)

vis = Visualizer()

loader = get_dataloader()

for epoch in range(settings.epochs):
    loss_G = 0
    loss_D = 0
    for i, (img, label) in enumerate(loader): 
        latent_point.normal_()

        fake = G(latent_point)
        pred_fake = D(fake)

        if i%2==1:
            loss_G = torch.mean(-torch.log(pred_fake))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
        else:
            pred_real = D(img)
            loss_D = torch.mean(-torch.log(1 - pred_fake) - torch.log(pred_real))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        if i % 100 == 99:
            print("Loss D: {}, Loss G: {}".format(loss_D, loss_G), end="\r")
            vis.update_batch(img, "real")
            vis.update_batch(fake, "fake")

