import torch
from utils.data import *
from utils.losses import phase_error, total_variation, speckle_brightness, lag_one_coherence, coherence_factor
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
from utils.plot import imagesc, plot_loss
from scripts.das import das
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def dbua(sample, loss_name, c_init_assumed):

    name = getName(sample, loss_name, N_ITERS, Z_GROWING, RANDOM_PATCHING)

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
        return time_of_flight(xe, ze, xi, zi, model, fnum=0.5, npts=NPTS_IMAGE, Dmin=3e-3, mode=1)

    def makeImage(model):
        t = tof_image(model)
        return torch.abs(das(iqdata, t - t0, t, fs, fd))

    def tof_patch(model):
        return time_of_flight(xe, ze, xp, zp, model, fnum=0.5, npts=NPTS_PATCH, Dmin=3e-3, mode=0)

    # Define loss functions
    def loss_wrapper(func, model):
        t = tof_patch(model)
        return func(iqdata, t - t0, t, fs, fd)

    sb_loss = lambda model: 1 - loss_wrapper(speckle_brightness, model)
    lc_loss = lambda model: 1 - torch.mean(loss_wrapper(lag_one_coherence, model))
    cf_loss = lambda model: 1 - torch.mean(loss_wrapper(coherence_factor, model))

    def pe_loss(model, z_max = PHASE_ERROR_Z_MAX):
        # Uniform patching
        xpc = torch.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP, device=device)
        zpc = torch.linspace(PHASE_ERROR_Z_MIN, z_max, NZP, device=device)
        dxpc, dzpc = xpc[1] - xpc[0] - (NXK - 1) * wl0 / 2, zpc[1] - zpc[0] - (NZK - 1) * wl0 / 2
        xpc, zpc = torch.meshgrid(xpc, zpc, indexing="ij")

        if RANDOM_PATCHING:  # Random patching
            minval = torch.tensor([-dxpc / 2, -dzpc / 2])
            maxval = torch.tensor([+dxpc / 2, +dzpc / 2])
            offsets = torch.rand(N_PATCHES, 2) * (maxval - minval) + minval
            offsets = offsets.to(device)
            xpc = xpc.flatten() + offsets[:, 0]
            zpc = zpc.flatten() + offsets[:, 1]

        xpp = xpc.reshape(1, -1, 1) + xk.reshape(1, 1, -1)
        zpp = zpc.reshape(1, -1, 1) + xk.reshape(1, 1, -1)
        xpp = xpp + 0 * zpp  # Manual broadcasting
        zpp = zpp + 0 * xpp  # Manual broadcasting

        t = time_of_flight(xe, ze, xpp, zpp, model, fnum=0.5, npts=NPTS_PATCH, Dmin=3e-3, mode=0)
        dphi = phase_error(iqdata, t - t0, t, fs, fd)
        valid = dphi != 0
        dphi = torch.where(valid, dphi, torch.nan)
        return torch.nanmean(torch.log1p(torch.square(100 * dphi)))

    tv = lambda c: total_variation(c) * dxc * dzc

    def loss(c, model, z_max):
        if loss_name == "sb":  # Speckle brightness
            return sb_loss(model) + tv(c) * LAMBDA_TV
        elif loss_name == "lc":  # Lag one coherence
            return lc_loss(model) + tv(c) * LAMBDA_TV
        elif loss_name == "cf":  # Coherence factor
            return cf_loss(model) + tv(c) * LAMBDA_TV
        elif loss_name == "pe":  # Phase error
            return pe_loss(model, z_max) + tv(c) * LAMBDA_TV
        else:
            assert False

    fig, _ = plt.subplots(1, 2, figsize=[9, 4])
    # Create the image axes for plotting
    ximm = xi[:, 0] * 1e3
    zimm = zi[0, :] * 1e3
    xcmm = xc * 1e3
    zcmm = zc * 1e3
    bdr = [-45, +5]
    cdr = np.array([-50, +50]) + \
          CTRUE[sample] if CTRUE[sample] > 0 else [1400, 1600]
    cmap = "seismic" if CTRUE[sample] > 0 else "jet"

    if MAKE_VIDEO:
        # Create the figure writer
        vobj = FFMpegWriter(fps=30)
        vobj.setup(fig, "videos/%s_opt%s.mp4" % (sample, loss_name), dpi=144)

    # Create a nice figure on first call, update on subsequent calls
    @torch.no_grad()
    def makeFigure(model, i, cimg, handles=None):
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
        losses = (sb_loss(model).item(), cf_loss(model).item(),
                  pe_loss(model).item(), tv(c).item() * LAMBDA_TV)
        if handles is None:
            # On the first time, create the figure
            fig.clf()
            plt.subplot(121)
            hbi = imagesc(ximm.cpu(), zimm.cpu(), bimg.cpu(), bdr, cmap="gray", interpolation="bicubic")
            hbt = plt.title("SB: %.2f, CF: %.3f, PE: %.3f, TV:%.3f" % losses)

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
            hbt.set_text("SB: %.2f, CF: %.3f, PE: %.3f, TV:%.3f" % losses)
            if CTRUE[sample] > 0:
                hct.set_text("Iteration %d: MAE %.2f" % (i, torch.mean(torch.abs(cimg - CTRUE[sample]))))
            else:
                hct.set_text("Iteration %d: Mean value %.2f" % (i, torch.mean(cimg)))

        plt.savefig(f"scratch/{name}.png")

    # Initial survey of losses vs. global sound speed
    if c_init_assumed == 0: c_init_assumed = ASSUMED_C
    c = c_init_assumed * torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)).to(device)
    c_normalized = normalize(c)

    # Create the optimizer
    xcm, zcm = torch.meshgrid(xc, zc, indexing="ij")
    coords = torch.stack([xcm, zcm], dim=2).reshape(-1, 2)
    dataloader = data.DataLoader(CoordDataset(coords), batch_size=256, shuffle=True)

    l1_loss = torch.nn.L1Loss()
    model = Model("./utils/config.json", 2, 1)
    optimizer1 = torch.optim.Adam(params=model.parameters(), lr=0.001, amsgrad=True)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=500)

    # Initializing Sound Speed
    with tqdm(range(500), desc="Initializing Sound Speed", unit="iter") as pbar:
        for _ in pbar:
            model.train()
            total_loss = 0
            for batch_coords in dataloader:
                batch_coords = batch_coords.to(device=device)
                c_pre = model(batch_coords).reshape(-1, 1)
                loss_value1 = l1_loss(c_pre, c_normalized.reshape(-1, 1)[:c_pre.shape[0]])
                total_loss += loss_value1.item()
                optimizer1.zero_grad()
                loss_value1.backward()
                optimizer1.step()
            scheduler1.step()
            pbar.set_postfix(loss=total_loss / len(dataloader), lr=optimizer1.param_groups[0]["lr"])

    c_init = denormalize(model(coords).detach()).reshape(SOUND_SPEED_NXC, SOUND_SPEED_NZC)

    if MAKE_FIGURE:
        print("Initializing First Frame")
        handles = makeFigure(model, 0, c_init)
        plt.savefig(f"scratch/{name}_init_{time.time()}.png")

    if MAKE_VIDEO:
        vobj.grab_frame()

    optimizer2 = torch.optim.Adam(params=model.parameters(), lr=0.005, amsgrad=True)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)
    l = list()

    # Optimization Loop
    with tqdm(range(N_ITERS), desc="Optimization Loop", unit="iter") as pbar:
        for i in pbar:
            z_max = PHASE_ERROR_Z_MIN + (PHASE_ERROR_Z_MAX - PHASE_ERROR_Z_MIN) \
                    * np.clip(round(i / (N_ITERS * 0.8) / 0.125) * 0.125, 0.2,
                              1.0) if Z_GROWING else PHASE_ERROR_Z_MAX

            model.train()
            c_pre = model(coords).reshape(SOUND_SPEED_NXC, SOUND_SPEED_NZC)

            loss_value2 = loss(denormalize(c_pre), model, z_max)
            pbar.set_postfix(loss=loss_value2.item(), lr=optimizer2.param_groups[0]["lr"],
                             zmax=z_max)
            l.append(loss_value2.item())

            optimizer2.zero_grad()
            loss_value2.backward()
            optimizer2.step()
            scheduler2.step()

            if MAKE_VIDEO:
                makeFigure(model, i + 1, denormalize(c_pre.detach()), handles)
                vobj.grab_frame()  # Add to video writer

    if MAKE_VIDEO: vobj.finish()  # Close video writer
    if MAKE_FIGURE:
        print("Creating Final Frame")
        makeFigure(model, N_ITERS, denormalize(c_pre.detach()))
        plt.savefig(f"scratch/{name}_finish_{time.time()}.png")

    plot_loss(l, sample)