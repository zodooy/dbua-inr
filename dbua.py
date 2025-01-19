import torch
from utils.data import *
from utils.losses import (
    phase_error,
    total_variation,
)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch
from scripts.tof import time_of_flight
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from torch.utils import data
from utils.inr import Model
from utils.plot import imagesc
from scripts.das import das

device = "cuda" if torch.cuda.is_available() else "cpu"
MAKE_VIDEO = False

def dbua(sample, loss_name):
    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(sample)
    xe, _, ze = np.array(elpos)
    iqdata = torch.tensor(iqdata, device=device)
    xe = torch.tensor(xe, device=device)
    ze = torch.tensor(ze, device=device)
    t0 = torch.tensor(t0, device=device)
    wl0 = ASSUMED_C / fd  # wavelength (λ)

    # B-mode image dimensions
    xi = torch.arange(BMODE_X_MIN, BMODE_X_MAX, wl0 / 3)
    zi = torch.arange(BMODE_Z_MIN, BMODE_Z_MAX, wl0 / 3)
    nxi, nzi = xi.size(0), zi.size(0)
    xi, zi = torch.meshgrid(xi, zi, indexing="ij")
    xi, zi = xi.to(device), zi.to(device)

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
    def tof_image(model):
        return time_of_flight(xe, ze, xi, zi, model, fnum=0.5, npts=64, Dmin=3e-3, mode=1)

    def makeImage(model):
        t = tof_image(model)
        return torch.abs(das(iqdata, t - t0, t, fs, fd))

    def tof_patch(model):
        return time_of_flight(xe, ze, xp, zp, model, fnum=0.5, npts=25, Dmin=3e-3, mode=0)

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

    if MAKE_VIDEO:
        # Create the figure writer
        fig, _ = plt.subplots(1, 2, figsize=[9, 4])
        vobj = FFMpegWriter(fps=30)
        vobj.setup(fig, "videos/%s_opt%s.mp4" % (sample, loss_name), dpi=144)

        # Create the image axes for plotting
        ximm = xi[:, 0] * 1e3
        zimm = zi[0, :] * 1e3
        xcmm = xc * 1e3
        zcmm = zc * 1e3
        bdr = [-45, +5]
        cdr = np.array([-50, +50]) + \
              CTRUE[sample] if CTRUE[sample] > 0 else [1400, 1600]
        cmap = "seismic" if CTRUE[sample] > 0 else "jet"

    # Create a nice figure on first call, update on subsequent calls
    @torch.no_grad()
    def makeFigure(cimg, i, handles=None, pbar=None):
        b = makeImage(model)
        if handles is None:
            bmax = torch.max(b)
        else:
            hbi, hci, hbt, hct, bmax = handles
        bimg = b / bmax
        bimg = bimg + 1e-10 * (bimg == 0)  # Avoid nans
        bimg = 20 * torch.log10(bimg)
        bimg = torch.reshape(bimg, (nxi, nzi)).T
        cimg = torch.reshape(cimg, (SOUND_SPEED_NXC, SOUND_SPEED_NZC)).T
        losses = (sb_lc_cf_loss("speckle_brightness", model).item(),
                  sb_lc_cf_loss("coherence_factor", model).item(),
                  pe_loss(c).item(), tv(c).item() * LAMBDA_TV)
        if handles is None:
            # On the first time, create the figure
            fig.clf()
            plt.subplot(121)
            hbi = imagesc(ximm.cpu(), zimm.cpu(), bimg.cpu(), bdr, cmap="gray", interpolation="bicubic")
            hbt = plt.title("SB: %.2f, CF: %.3f, PE: %.3f, TV: %.3f" % losses)

            plt.xlim(ximm[0].cpu(), ximm[-1].cpu())
            plt.ylim(zimm[-1].cpu(), zimm[0].cpu())
            plt.subplot(122)
            hci = imagesc(xcmm.cpu(), zcmm.cpu(), cimg.cpu(), cdr, cmap=cmap, interpolation="bicubic")
            if CTRUE[sample] > 0:  # When ground truth is provided, show the error
                hct = plt.title("Iteration %d: MAE %.2f" % (i, torch.mean(torch.abs(cimg - CTRUE[sample]))))
            else:
                hct = plt.title("Iteration %d: Mean value %.2f" % (i, torch.mean(cimg)))

            plt.xlim(ximm[0].cpu(), ximm[-1].cpu())
            plt.ylim(zimm[-1].cpu(), zimm[0].cpu())
            fig.tight_layout()
            return hbi, hci, hbt, hct, bmax
        else:
            hbi.set_data(bimg.cpu())
            hci.set_data(cimg.cpu())
            hbt.set_text("SB: %.2f, CF: %.3f, PE: %.3f, TV: %.3f" % losses)
            if CTRUE[sample] > 0:
                hct.set_text("Iteration %d: MAE %.2f" % (i, torch.mean(torch.abs(cimg - CTRUE[sample]))))
            else:
                hct.set_text("Iteration %d: Mean value %.2f" % (i, torch.mean(cimg)))

        if pbar: pbar.set_postfix(sb=losses[0], cf=losses[1], pe=losses[2], tv=losses[3])
        plt.savefig(f"scratch/{sample}.png")

    # Initial survey of losses vs. global sound speed
    c = ASSUMED_C * torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)).to(device)

    # Create the optimizer
    xcm, zcm = torch.meshgrid(xc, zc, indexing="ij")
    coords = torch.stack([xcm, zcm], dim=2).reshape(-1, 2)
    dataloader = data.DataLoader(CoordDataset(coords), batch_size=256, shuffle=True)

    l1_loss = torch.nn.L1Loss()
    model = Model("./utils/config.json", 2, 1)
    optimizer1 = torch.optim.Adam(params=model.parameters(), lr=0.001, amsgrad=True)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=500)

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
                optimizer1.zero_grad()
                loss_value1.backward()
                optimizer1.step()
            scheduler1.step()
            pbar.set_postfix(loss=total_loss / len(dataloader), lr=optimizer1.param_groups[0]["lr"])


    if MAKE_VIDEO:
        handles = makeFigure(model, 0)
        plt.savefig(f"scratch/{sample}_init.png")
        vobj.grab_frame()

    optimizer2 = torch.optim.Adam(params=model.parameters(), lr=0.001, amsgrad=True)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)

    # Optimization Loop
    with tqdm(range(N_ITERS), desc="Optimization Loop", unit="iter") as pbar:
        for i in pbar:
            model.train()
            c_pre = model(coords).reshape(SOUND_SPEED_NXC, SOUND_SPEED_NZC)
            loss_value2 = loss(c_pre, model)
            pbar.set_postfix(loss=loss_value2.item(), lr=optimizer2.param_groups[0]["lr"])
            optimizer2.zero_grad()
            loss_value2.backward()
            optimizer2.step()
            scheduler2.step()
            if MAKE_VIDEO:
                makeFigure(c, i + 1, handles, pbar)
                vobj.grab_frame()  # Add to video writer

    if MAKE_VIDEO:  vobj.finish()  # Close video writer
    makeFigure(c, N_ITERS)


if __name__ == "__main__":
    dbua(SAMPLE, LOSS)