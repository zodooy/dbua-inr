from torch.func import vmap
from utils.interp import *

def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):

    # 默认的线性组合是对所有元素求和
    if A is None:
        A = torch.ones(iqraw.shape[0], device=iqraw.device)
    if B is None:
        B = torch.ones(iqraw.shape[1], device=iqraw.device)

    # 选择插值函数
    fints = {
        "nearest": interp_nearest,
        "linear": interp_linear,
        "cubic": interp_cubic,
        "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
        "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
    }
    fint = fints[interp]

    # 基带插值，将iq数据按照时间延迟t进行插值
    def bbint(iq, t):
        iqfoc = fint(iq, fs * t) # 插值时间通过fs * t转换为采样点位置
        return iqfoc * torch.exp(2j * torch.pi * fd * t) # 对插值后的 IQ 数据进行基带调制

    # 内层函数：处理单个接收通道
    def das_b(x):
        iq_i, tA_i = x
        return torch.tensordot(B.to(torch.complex128), vmap(bbint)(iq_i, tA_i + tB) * apoB, dims=([-1], [0]))

    # 外层函数：处理多个发射通道
    return torch.tensordot(A.to(torch.complex128), torch.stack([das_b(x) for x in zip(iqraw, tA)]) * apoA, dims=([-1], [0]))

    # # 提高并行程度以加速，但可能会占用更多资源
    # # 内层函数：处理单个接收通道
    # def das_b(iq_i, tA_i):
    #     return torch.tensordot(B.to(torch.complex128), vmap(bbint)(iq_i, tA_i + tB) * apoB, dims=([-1], [0]))
    #
    # # 外层函数：处理多个发射通道
    # return torch.tensordot(A.to(torch.complex128), vmap(bbint)(iqraw, tA) * apoA, dims=([-1], [0]))



