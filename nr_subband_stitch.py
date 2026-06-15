#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G NR 大带宽子带拼接仿真
=========================

场景：载波带宽 100MHz，SCS=30kHz，273 RB -> 3276 个有效子载波(RE)。
ADC 采样率 Fs=122.88MHz，一个 OFDM 符号(去 CP)= 4096 个时域点。

流程：
  1) 用 3276 个频域子载波 -> 映射到 4096 网格 -> IFFT 造出"ADC 时域采样"。
  2) 路径 A(全带，黄金参考)：直接对 4096 时域点做 4096-FFT，抽取 3276 个有效子载波。
  3) 路径 B(子带拼接，受芯片能力限制)：把 100MHz 切成 5 个子带
     (RB=[55,54,55,54,55])，按"中射频(IF) + 基带(BB)"两组件分工:
        - 中射频: DDC -> 数字低通滤波(循环卷积) -> 下采样 4x
                  -> 施加"实际传递时延"(IF->BB 传数据的延迟, FFT 前时域模拟);
        - 基带  : 1024-FFT -> 相位补偿(FIR 群延迟 + "补偿时延") -> 拼接成全带 3276。
     每个子带可独立配置 actual_delay(实际时延)与 comp_delay(补偿时延);
     当二者一致时, 时延引入的相位偏差被完全补偿, 拼接结果与全带处理差异很小。
  4) 比较 A 与 B 的最大/平均 幅度差异 与 相位差异(补偿后为主, 同时给出未补偿原始值)。

对随机 QPSK 与全 1 平坦谱各跑一次(实际时延=补偿时延)，并额外做一次"时延失配"
对照(补偿时延偏离 0.5 样点, 相位差明显变大)。打印指标并保存 matplotlib 图。
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# 1. 系统常量与子带划分
# ----------------------------------------------------------------------------
FS = 122.88e6            # ADC 采样率
SCS = 30e3              # 子载波间隔
NFFT = 4096            # 全带 IFFT/FFT 点数(一个符号去 CP 的时域点数)
N_SC = 3276            # 有效子载波数 = 273 RB * 12
RB_LIST = [55, 54, 55, 54, 55]   # 5 个子带的 RB 数, 合计 273
D = 4                  # 下采样因子: 122.88MHz / 4 = 30.72MHz
NSUB = NFFT // D       # 子带 FFT 点数 = 1024 (30.72MHz/1024 = 30kHz, 每 bin 一个子载波)
DC_SC = N_SC // 2      # = 1638, 频率为 0 的子载波索引, f(s)=(s-DC_SC)*SCS
GRID_OFFSET = (NFFT - N_SC) // 2   # = 410, 有效子载波在 fftshift 网格中的起始位置
QPSK_SEED = 20240601

# 每个子带"中射频 -> 基带"的实际传递时延(下采样域样点, 可为小数)。
# 模拟 5 个子带各自不同的传递延迟; 应用于下采样后、FFT 前的时域。
ACTUAL_DELAYS = [0.0, 1.0, 2.5, 3.0, 4.5]
# 每个子带在基带侧用于频域相位补偿的补偿时延(下采样域样点)。
# 与 ACTUAL_DELAYS 一致时, 时延引入的相位偏差被完全补偿。
COMP_DELAYS = [0.0, 1.0, 2.5, 3.0, 4.5]

assert sum(RB_LIST) == 273
assert N_SC == 273 * 12
assert len(ACTUAL_DELAYS) == len(RB_LIST) == len(COMP_DELAYS)


@dataclass
class Subband:
    """单个子带的配置与划分。

    s_lo, s_hi  : 子载波索引区间(闭)。
    center      : 整数中心子载波索引(DDC 频率为 SCS 整数倍, 搬移后落在整数 bin)。
    actual_delay: 中射频->基带的实际传递时延(下采样域样点), 在 FFT 前的时域模拟。
    comp_delay  : 基带频域相位补偿所用的补偿时延(下采样域样点)。
    """
    s_lo: int
    s_hi: int
    center: int
    actual_delay: float = 0.0
    comp_delay: float = 0.0


def subband_layout(actual_delays=ACTUAL_DELAYS, comp_delays=COMP_DELAYS):
    """返回每个子带的 Subband 配置。

    center 取整数, 保证 DDC 频率是 SCS 的整数倍, 从而搬移后子载波恰好落在
    1024-FFT 的整数 bin 上, 实现无 bin 间泄漏的干净拼接。
    """
    layout = []
    s = 0
    for rb, da, dc in zip(RB_LIST, actual_delays, comp_delays):
        n = rb * 12
        s_lo, s_hi = s, s + n - 1
        center = int(round((s_lo + s_hi) / 2.0))
        layout.append(Subband(s_lo, s_hi, center, da, dc))
        s += n
    return layout


SUBBANDS = subband_layout()


# ----------------------------------------------------------------------------
# 2. 频域数据构造
# ----------------------------------------------------------------------------
def build_symbol_freq(mode):
    """构造一个符号的 3276 个频域复数。

    mode='qpsk' : 固定种子的随机 QPSK (单位幅度)。
    mode='ones' : 全 1+0j 平坦谱, 最直观暴露滤波器通带纹波/边缘滚降。
    """
    if mode == "qpsk":
        rng = np.random.default_rng(QPSK_SEED)
        bits = rng.integers(0, 2, size=(N_SC, 2))
        sym = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
        return sym / np.sqrt(2)
    elif mode == "ones":
        return np.ones(N_SC, dtype=complex)
    raise ValueError(f"unknown mode: {mode}")


def map_to_grid(freq3276):
    """3276 个有效子载波 -> 4096 频域网格 (中心放置, fftshift 约定)。"""
    grid_shift = np.zeros(NFFT, dtype=complex)
    grid_shift[GRID_OFFSET:GRID_OFFSET + N_SC] = freq3276
    return np.fft.ifftshift(grid_shift)   # 转成标准 FFT bin 顺序


def make_time(freq3276):
    """频域 -> 4096 时域点 (ADC 时域输出)。"""
    grid = map_to_grid(freq3276)
    return np.fft.ifft(grid)


# ----------------------------------------------------------------------------
# 3. 路径 A: 全带 4096-FFT 参考
# ----------------------------------------------------------------------------
def fullband_demod(time_sig):
    """对 4096 时域点做 4096-FFT, 抽取中心 3276 个有效子载波。"""
    F = np.fft.fftshift(np.fft.fft(time_sig))
    return F[GRID_OFFSET:GRID_OFFSET + N_SC]


# ----------------------------------------------------------------------------
# 4. 子带抽取低通滤波器
# ----------------------------------------------------------------------------
def design_lpf():
    """线性相位 FIR 低通 (Kaiser 窗)。

    - 通带需平到子带半宽 660 SC -> ±9.9MHz, 取 fp=10.2MHz 留余量;
    - 仅当频率 > ~20.8MHz (bin 694) 时, 下采样后才会混叠回本子带 bin,
      故阻带 fst=16MHz 已足够安全, 过渡带很宽;
    - 阻带衰减目标 80dB。返回奇数长度 taps, 整数群延迟 g=(taps-1)/2。
    """
    fp, fst, atten_db = 10.2e6, 16.0e6, 80.0
    numtaps, beta = signal.kaiserord(atten_db, (fst - fp) / (FS / 2.0))
    if numtaps % 2 == 0:
        numtaps += 1
    cutoff = (fp + fst) / 2.0
    h = signal.firwin(numtaps, cutoff, window=("kaiser", beta), fs=FS)
    return h


# ----------------------------------------------------------------------------
# 5. 路径 B: 子带 DDC -> 滤波 -> 下采样 -> 1024-FFT -> 拼接
# ----------------------------------------------------------------------------
def circular_filter(x, h):
    """对周期信号 x 施加 FIR h 的循环卷积 (FFT 实现), 消除块边缘暂态。"""
    H = np.fft.fft(h, len(x))
    return np.fft.ifft(np.fft.fft(x) * H)


def apply_circular_delay(x, d):
    """对周期信号 x 施加 d 个样点的循环时延(d>0 表示延迟/右移, 支持小数)。

    用频域线性相位实现带限循环时延; 整数 d 时等价于 np.roll(x, d)。
    """
    if d == 0:
        return x
    n = len(x)
    f = np.fft.fftfreq(n) * n   # [0,1,...,n/2-1,-n/2,...,-1], 即各 bin 的有符号频率
    return np.fft.ifft(np.fft.fft(x) * np.exp(-1j * 2 * np.pi * f * d / n))


def subband_demod(time_sig, h, subbands=SUBBANDS):
    """子带拼接解调。返回 (stitched_comp, stitched_raw)。

    中射频(IF): DDC -> 数字低通 -> 下采样 -> 施加"实际传递时延"(FFT 前时域)。
    基带(BB)  : 1024-FFT -> 相位补偿(FIR 群延迟 + 补偿时延) -> 拼接。

    stitched_raw : 未做任何补偿(含 FIR 群延迟与传递时延), 相位差最大;
    stitched_comp: 补偿 FIR 群延迟 + 子带配置的补偿时延后的结果(主指标)。
                   当某子带 comp_delay == actual_delay 时, 时延相位偏差被完全抵消。
    """
    g = (len(h) - 1) / 2.0                    # FIR 群延迟(全速率样点)
    n = np.arange(NFFT)
    stitched_raw = np.zeros(N_SC, dtype=complex)
    stitched_comp = np.zeros(N_SC, dtype=complex)

    for sb in subbands:
        c = sb.center
        # --- [IF] DDC: 把子带中心搬到基带(频率为 SCS 整数倍, 循环友好) ---
        ddc = np.exp(-1j * 2 * np.pi * (c - DC_SC) * n / NFFT)
        x = time_sig * ddc

        # --- [IF] 数字低通(循环卷积) ---
        y = circular_filter(x, h)

        # --- [IF] 下采样 4x -> 1024 点 ---
        y_dec = y[::D]

        # --- [IF] 中射频->基带 传递时延(下采样域, FFT 前时域模拟) ---
        y_dec = apply_circular_delay(y_dec, sb.actual_delay)

        # --- [BB] 1024-FFT 解调, 取本子带有效子载波 ---
        Fsub = np.fft.fft(y_dec)
        for s in range(sb.s_lo, sb.s_hi + 1):
            m = s - c                         # 子载波有符号偏移(=有符号频率 bin)
            k = m % NSUB                      # 子载波 s 落在的 1024-FFT bin
            val = D * Fsub[k]                 # 下采样带来的 1/D 幅度归一化
            stitched_raw[s] = val
            # [BB] 相位补偿:
            #   FIR 群延迟  -> exp(+j2π·m·g/NFFT)         (全速率域, 用 NFFT)
            #   补偿时延    -> exp(+j2π·m·comp_delay/NSUB) (下采样域, 用 NSUB)
            comp = (np.exp(1j * 2 * np.pi * m * g / NFFT)
                    * np.exp(1j * 2 * np.pi * m * sb.comp_delay / NSUB))
            stitched_comp[s] = val * comp

    return stitched_comp, stitched_raw


# ----------------------------------------------------------------------------
# 6. 指标
# ----------------------------------------------------------------------------
def compute_metrics(ref, test):
    """返回幅度(dB)与相位(deg)差异的 max(|.|) 与 mean(|.|)。"""
    amp_db = 20.0 * np.log10(np.abs(test) / np.abs(ref))
    phase_deg = np.degrees(np.angle(test * np.conj(ref)))
    return {
        "amp_max_db": np.max(np.abs(amp_db)),
        "amp_mean_db": np.mean(np.abs(amp_db)),
        "phase_max_deg": np.max(np.abs(phase_deg)),
        "phase_mean_deg": np.mean(np.abs(phase_deg)),
        "_amp_db": amp_db,
        "_phase_deg": phase_deg,
    }


def print_metrics(title, m):
    print(f"  [{title}]")
    print(f"    幅度差异: max={m['amp_max_db']:.4e} dB,  mean={m['amp_mean_db']:.4e} dB")
    print(f"    相位差异: max={m['phase_max_deg']:.4e} deg, mean={m['phase_mean_deg']:.4e} deg")


# ----------------------------------------------------------------------------
# 7. 绘图
# ----------------------------------------------------------------------------
def plot_results(mode, ref, comp, m_comp, m_raw, subbands=SUBBANDS):
    sc = np.arange(N_SC)
    bounds = [sb.s_lo for sb in subbands] + [N_SC]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.suptitle(f"NR subband-stitch vs full-band 4096-FFT  ({mode})", fontsize=13)

    def mark_bands(ax):
        for b in bounds:
            ax.axvline(b, color="gray", ls="--", lw=0.6, alpha=0.6)

    # 图1: 幅度差(dB)
    ax = axes[0]
    ax.plot(sc, m_raw["_amp_db"], lw=0.6, alpha=0.5, label="raw")
    ax.plot(sc, m_comp["_amp_db"], lw=0.6, label="group-delay compensated")
    mark_bands(ax)
    ax.set_ylabel("amplitude diff (dB)")
    ax.set_title("amplitude difference (subband - fullband)")
    ax.legend(loc="upper right", fontsize=8)

    # 图2: 相位差(deg)
    ax = axes[1]
    ax.plot(sc, m_raw["_phase_deg"], lw=0.6, alpha=0.5, label="raw")
    ax.plot(sc, m_comp["_phase_deg"], lw=0.6, label="group-delay compensated")
    mark_bands(ax)
    ax.set_ylabel("phase diff (deg)")
    ax.set_title("phase difference (subband - fullband)")
    ax.legend(loc="upper right", fontsize=8)

    # 图3: 幅度谱叠加
    ax = axes[2]
    ax.plot(sc, 20 * np.log10(np.abs(ref) + 1e-12), lw=0.6, label="fullband ref")
    ax.plot(sc, 20 * np.log10(np.abs(comp) + 1e-12), lw=0.6, alpha=0.7, label="subband stitched")
    mark_bands(ax)
    ax.set_ylabel("amplitude (dB)")
    ax.set_xlabel("subcarrier index")
    ax.set_title("amplitude spectra overlay (dashed = subband edges)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"result_{mode}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"    已保存图: {out}")


# ----------------------------------------------------------------------------
# 8. 主流程
# ----------------------------------------------------------------------------
def run(mode, h, subbands=SUBBANDS, save_plot=True, tag=None):
    label = mode if tag is None else f"{mode} [{tag}]"
    print(f"\n================ 测试信号: {label} ================")
    freq = build_symbol_freq(mode)
    time_sig = make_time(freq)

    # 路径 A: 全带参考(并自检 IFFT/FFT/映射 正确性)
    ref = fullband_demod(time_sig)
    selfcheck = np.max(np.abs(ref - freq))
    print(f"  自检(全带参考 vs 输入)最大误差: {selfcheck:.3e}")

    # 路径 B: 子带拼接
    comp, raw = subband_demod(time_sig, h, subbands)

    m_comp = compute_metrics(ref, comp)
    m_raw = compute_metrics(ref, raw)
    print("  子带拼接 vs 全带:")
    print_metrics("补偿后 (FIR群延迟+补偿时延, 主指标)", m_comp)
    print_metrics("原始 (未补偿)", m_raw)

    if save_plot:
        plot_results(mode, ref, comp, m_comp, m_raw, subbands)
    return m_comp


def main():
    h = design_lpf()
    print(f"子带低通 FIR: taps={len(h)}, 群延迟 g={(len(h)-1)//2} 全速率样点")
    print("子带配置(s_lo, s_hi, center, 实际时延, 补偿时延; 时延单位=下采样域样点):")
    for sb in SUBBANDS:
        print(f"  {sb.s_lo:>4}..{sb.s_hi:<4} center={sb.center:<4} "
              f"actual_delay={sb.actual_delay:<4} comp_delay={sb.comp_delay}")

    # 实际时延 == 补偿时延 -> 时延相位偏差被完全补偿, 差异应较小。
    for mode in ("qpsk", "ones"):
        run(mode, h)

    # 对照演示: 故意让补偿时延偏离实际时延 0.5 个样点 -> 相位差应明显变大。
    print("\n######## 对照: 补偿时延失配(comp = actual + 0.5 样点) ########")
    mismatched = subband_layout(
        actual_delays=ACTUAL_DELAYS,
        comp_delays=[d + 0.5 for d in ACTUAL_DELAYS],
    )
    run("qpsk", h, subbands=mismatched, save_plot=False, tag="时延失配")


if __name__ == "__main__":
    main()
