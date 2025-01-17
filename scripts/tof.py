import torch


def time_of_flight(x0, z0, x1, z1, model, fnum: float, npts: int, Dmin: float):
    # define the range of t
    t_all = torch.linspace(1, 0, npts + 1, device=x0.device)[:-1].flip(0)

    # calculate tof
    dx = torch.abs(x1 - x0)
    dz = torch.abs(z1 - z0)
    dtrue = torch.sqrt(dx ** 2 + dz ** 2)

    def get_slowness(t):
        xt = t * (x1 - x0) + x0
        zt = t * (z1 - z0) + z0

        coords = torch.stack([xt, zt], dim=3).reshape(-1, 2)

        sos = model(coords).detach().reshape(xt.shape)

        return 1 / sos

    slowness = torch.stack([get_slowness(t) for t in t_all])
    tof = torch.nanmean(slowness, dim=0) * dtrue

    # filter valid tof
    fnum_valid = torch.abs(2 * fnum * dx) <= dz
    Dmin_valid = torch.logical_and(dz < Dmin * fnum, dx < Dmin / 2)
    valid = torch.logical_or(fnum_valid, Dmin_valid)
    tof_valid = torch.where(valid, tof, torch.tensor(1.0, device=tof.device))
    tof = torch.where(valid, tof_valid, torch.tensor(-10.0, device=tof.device))

    return tof
