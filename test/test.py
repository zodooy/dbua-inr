import numpy as np
import torch
from tqdm import tqdm

from scripts.tof import time_of_flight
from utils.data import *
from utils.losses import (
    phase_error,
    total_variation,
)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from torch.utils import data
from utils.inr import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

def dbua(sample, loss_name):
    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(sample)
    xe, _, ze = np.array(elpos)
    iqdata = torch.tensor(iqdata, device=device)
    xe = torch.tensor(xe, device=device)
    ze = torch.tensor(ze, device=device)
    t0 = torch.tensor(t0, device=device)
    wl0 = ASSUMED_C / fd  # wavelength (λ)

    # Sound speed grid dimensions
    xc = torch.linspace(SOUND_SPEED_X_MIN, SOUND_SPEED_X_MAX, SOUND_SPEED_NXC, device=device)
    zc = torch.linspace(SOUND_SPEED_Z_MIN, SOUND_SPEED_Z_MAX, SOUND_SPEED_NZC, device=device)
    dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]

    # Kernels to use for loss calculations (2λ x 2λ patches)
    xk, zk = torch.meshgrid((torch.arange(NXK) - (NXK - 1) / 2) * wl0 / 2,
                            (torch.arange(NZK) - (NZK - 1) / 2) * wl0 / 2, indexing="ij")
    xk, zk = xk.to(device), zk.to(device)

    # Kernel patch centers, distributed throughout the field of view
    xpc, zpc = torch.meshgrid(torch.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP),
                              torch.linspace(PHASE_ERROR_Z_MIN, PHASE_ERROR_Z_MAX, NZP), indexing="ij")
    xpc, zpc = xpc.to(device), zpc.to(device)

    # Explicit broadcasting. Dimensions will be [elements, pixels, patches]
    xe = torch.reshape(xe, (-1, 1, 1))
    ze = torch.reshape(ze, (-1, 1, 1))
    xp = torch.reshape(xpc, (1, -1, 1)) + torch.reshape(xk, (1, 1, -1))
    zp = torch.reshape(zpc, (1, -1, 1)) + torch.reshape(zk, (1, 1, -1))
    xp = xp + 0 * zp  # Manual broadcasting
    zp = zp + 0 * xp  # Manual broadcasting

    # Compute time-of-flight for each {image, patch} pixel to each element
    def tof_patch(model):
        return time_of_flight(xe, ze, xp, zp, model, fnum=0.5, npts=64, Dmin=3e-3)

    # Define loss functions
    def loss_wrapper(func, model):
        t = tof_patch(model)
        return func(iqdata, t - t0, t, fs, fd)

    def sb_lc_cf_loss(func, model):
        return 1 - loss_wrapper(func, model)

    def pe_loss(model):
        t = tof_patch(model)
        dphi = phase_error(iqdata, t - t0, t, fs, fd)
        valid = dphi != 0
        dphi = torch.where(valid, dphi, torch.nan)
        return torch.nanmean(torch.log1p(torch.square(100 * dphi)))

    tv = lambda c: total_variation(c) * dxc * dzc

    def loss(c, model):
        if loss_name == "sb":  # Speckle brightness
            return sb_lc_cf_loss("speckle_brightness", model) + tv(c) * LAMBDA_TV
        elif loss_name == "lc":  # Lag one coherence
            return torch.mean(sb_lc_cf_loss("lag_one_coherence", model)) + tv(c) * LAMBDA_TV
        elif loss_name == "cf":  # Coherence factor
            return torch.mean(sb_lc_cf_loss("coherence_factor", model)) + tv(c) * LAMBDA_TV
        elif loss_name == "pe":  # Phase error
            return pe_loss(model) + tv(c) * LAMBDA_TV
        else:
            assert False

    # Initial survey of losses vs. global sound speed
    c = ASSUMED_C * torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)).to(device)

    # Create the optimizer
    xcm, zcm = torch.meshgrid(xc, zc, indexing="ij")
    coords = torch.stack([xcm, zcm], dim=2).reshape(-1, 2)
    dataloader = data.DataLoader(CoordDataset(coords), batch_size=256, shuffle=True)

    l1_loss = torch.nn.L1Loss()
    model = Model("../utils/config.json", 2, 1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, amsgrad=True)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Initializing Sound Speed
    with tqdm(range(1000), desc="Initializing Sound Speed", unit="iter") as pbar:
        for _ in pbar:
            model.train()
            total_loss = 0
            for batch_coords in dataloader:
                batch_coords = batch_coords.to(device=device)
                c_pre = model(batch_coords).reshape(-1, 1)
                loss_value1 = l1_loss(c_pre, c.reshape(-1, 1)[:c_pre.shape[0]])
                total_loss += loss_value1.item()
                optimizer.zero_grad()
                loss_value1.backward()
                optimizer.step()
            scheduler1.step()
            pbar.set_postfix(loss=total_loss / len(dataloader), lr=optimizer.param_groups[0]["lr"])

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Optimization Loop
    with tqdm(range(N_ITERS), desc="Optimization Loop", unit="iter") as pbar:
        for _ in pbar:
            model.train()
            c_pre = model(coords).reshape(SOUND_SPEED_NXC, SOUND_SPEED_NZC)

            # define the range of t
            npts, fnum, Dmin = 64, 0.5, 3e-3
            t_all = torch.linspace(1, 0, npts + 1, device=xe.device)[:-1].flip(0)

            # calculate tof
            dx = torch.abs(xp - xe)
            dz = torch.abs(zp - ze)
            dtrue = torch.sqrt(dx ** 2 + dz ** 2)

            def get_slowness(t):
                print(t)
                xt = t * (xp - xe) + xe
                zt = t * (zp - ze) + ze

                coords = torch.stack([xt, zt], dim=3).reshape(-1, 2)

                sos = model(coords).reshape(xt.shape)

                return 1 / sos

            slowness = torch.stack([get_slowness(t) for t in t_all])
            tof = torch.nanmean(slowness, dim=0) * dtrue

            # filter valid tof
            fnum_valid = torch.abs(2 * fnum * dx) <= dz
            Dmin_valid = torch.logical_and(dz < Dmin * fnum, dx < Dmin / 2)
            valid = torch.logical_or(fnum_valid, Dmin_valid)
            tof_valid = torch.where(valid, tof, torch.tensor(1.0, device=tof.device))
            tof = torch.where(valid, tof_valid, torch.tensor(-10.0, device=tof.device))


            dphi = phase_error(iqdata, tof - t0, tof, fs, fd)
            valid = dphi != 0
            dphi = torch.where(valid, dphi, torch.nan)
            loss_pe = torch.nanmean(torch.log1p(torch.square(100 * dphi)))

            loss_value2 = loss_pe + tv(c_pre) * LAMBDA_TV





            pbar.set_postfix(loss=loss_value2.item(), lr=optimizer.param_groups[0]["lr"])
            optimizer.zero_grad()
            loss_value2.backward()
            optimizer.step()
            scheduler2.step()


if __name__ == "__main__":
    dbua(SAMPLE, LOSS)



