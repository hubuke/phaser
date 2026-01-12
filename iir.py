# First order IIR filter for multiple channels and profiles with one DSP and no blockram.
# DSP block with MSB aligned inputs and "round half down" rounding.
#
#
# Note: Migen translates the "out of range" pc mux selector to the last vaid mux input.

from ast import Constant
from migen import *

N_COEFF = 3  # [b0, b1, a0] number of coefficients for a first order iir

SHIFT_P = 11  # number of bits to right shift the dsp.p output for higher dynamic range

class Dsp(Module):
    def __init__(self):
        # xilinx dsp architecture (subset)
        self.a = a = Signal((25, True), reset_less=True)
        self.b = b = Signal((18, True), reset_less=True)
        self.c = c = Signal((48, True), reset_less=True)
        self.mux_p = mux_p = Signal()  # accumulator mux
        self.m = m = Signal((48, True), reset_less=True)
        self.p = p = Signal((48, True), reset_less=True)
        self.sync += [m.eq(a * b), p.eq(m + c), If(mux_p, p.eq(m + p))]

class Iir(Module):
    def __init__(self, w_coeff, w_data, log2_a0, n_profiles, n_channels):
        """
        w_coeff: 16 bits
        w_data: ADC_DATA_WIDTH = 16 bits
        log2_a0: log2 of a0 coefficient = 14 bits
        n_profiles: number of filter profiles = 4
        n_channels: number of channels = 2
        """
        # input strobe signal (start processing all channels)
        self.stb_in = stb_in = Signal()
        self.stb_out = stb_out = Signal()  # output strobe signal (all channels done)
        self.inp = inp = Array(Signal((w_data, True)) for _ in range(n_channels))
        self.outp = outp = Array(Signal((w_data, True)) for _ in range(n_channels))
        # coeff registers for all channels and profiles
        self.coeff = coeff = Array(
            Array(
                Array(Signal((w_coeff, True)) for _ in range(n_channels))
                for _ in range(n_profiles)
            )
            for _ in range(N_COEFF)
        )
        self.offset = offset = Array(
            Array(Signal((w_data, True)) for _ in range(n_channels))
            for _ in range(n_profiles)
        )
        # registers for selected profile for channel
        self.ch_profile = ch_profile = Array(
            Signal(max=n_profiles + 1) for _ in range(n_channels)
        )
        # output hold signal for each channel
        self.hold = hold = Array(Signal() for _ in range(n_channels))

        ###

        # Making these registers reset less results in worsend timing.
        # y1 register unique for each profile
        y1 = Array(
            Array(Signal((w_data, True)) for _ in range(n_channels))
            for _ in range(n_profiles)
        )
        # x0, x1 registers shared for all profiles
        x = Array(
            Array(Signal((w_data, True)) for _ in range(n_channels))
            for _ in range(N_COEFF - 1)
        )
        y0_clipped = Signal((w_data, True))
        profile_index = Signal(max=n_profiles + 1)
        channel_index = Signal(max=n_channels + 1)
        busy = Signal()
        # computation steps/pipeline:
        # 0    -> load coeff[0],xy[0]
        # 1    -> load coeff[1],xy[1], m0=coeff[0]*xy[0]
        # 2    -> load coeff[2],xy[2], m1=coeff[1]*xy[1], p0=offset+m0
        # 1(3) ->                      m2=coeff[2]*xy[2], p1=p0+m1
        # 2(4) ->                                         p2=p1+m2
        # 3(5) ->                                                      retrieve data y0=clip(p2)?hold
        step = Signal(2)  # computation step
        ch_profile_last_ch = Signal(max=n_profiles + 1)  # auxillary signal for muxing
        self.submodules.dsp = dsp = Dsp()
        assert w_data <= len(dsp.b)
        assert w_coeff <= len(dsp.a)
        overfloat_bits = len(dsp.p) - w_data - SHIFT_P

        self.sync += [
            # default to 0 and set to 1 further down if computation done in this cycle
            stb_out.eq(0),
            dsp.a.eq(coeff[step][profile_index][channel_index]),
            dsp.b.eq(
                x[channel_index][step]
            ),  # overwritten later if at step==2,
            dsp.c.eq(offset[profile_index][channel_index] << w_data),
            If(
                stb_in & ~busy,
                busy.eq(1),
                profile_index.eq(ch_profile[channel_index]),
                [xi[0].eq(i) for xi, i in zip(x, inp)],
            ),
            If(
                busy,
                step.eq(step + 1),
                If(step == 1, dsp.mux_p.eq(0)),
                If(
                    step == 2,
                    dsp.mux_p.eq(1),
                    step.eq(0),
                    channel_index.eq(channel_index + 1),
                    profile_index.eq(ch_profile[channel_index + 1]),
                    dsp.b.eq(y1[profile_index][channel_index]),
                    If(
                        (channel_index != 0)
                        & (channel_index != n_channels + 1)
                        & ~hold[channel_index - 1],
                        y1[ch_profile_last_ch][channel_index - 1].eq(y0_clipped),
                    ),
                ),
            ),
            # if done with all channels and last data is done
            If(
                (channel_index == n_channels) & (step == 2),
                channel_index.eq(0),
                profile_index.eq(ch_profile[0]),
                busy.eq(0),
                stb_out.eq(1),
                [xi[1].eq(xi[0]) for xi in x],
            ),
        ]
        self.comb += [
            # assign extra signal for xy adressing
            ch_profile_last_ch.eq(ch_profile[channel_index - 1]),
            [o.eq(y1[ch_profile[ch]][ch]) for ch, o in enumerate(outp)],
            # clipping to positive output range
            y0_clipped.eq(dsp.p >> SHIFT_P), 
            If(
                (dsp.p >> SHIFT_P) > (1 << (w_data - 1)) - 1,
                y0_clipped.eq((1 << (w_data - 1)) - 1), 
            ).Else(
                y0_clipped.eq(dsp.p >> SHIFT_P)
            ),
            If(dsp.p[-1] != 0, y0_clipped.eq(0)),
        ]
