import sionna.phy as sn
import numpy as np

class UncodedAWGN(sn.Block):
    def __init__ (self, num_bits_per_symbol=2):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.source = sn.mapping.BinarySource()
        self.constel = sn.mapping.Constellation("qam", num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constel)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constel)
        self.channel = sn.channel.AWGN()

    def call(self, batch_size, ebno_db):
        no = sn.utils.ebnodb2no(ebno_db, self.num_bits_per_symbol, coderate=1.0) #whats this?
        bits = self.source([batch_size, 1024]) # B x 1024
        x = self.mapper(bits) # B x 512
        y = self.channel(x, no) # B x 512
        llr = self.demapper(y, no) # B x 1024 
        return bits, llr

model = UncodedAWGN()

ber_plot = sn.utils.PlotBER()
ber_plot.simulate(
    model, 
    ebno_dbs=np.arange(-3, 9, 1),
    batch_size=4096,
    num_target_bit_errors=2000,
    max_mc_iter=100,
    soft_estimates=True,
    legend="Uncoded QPSK / AWGN",
    show_fig=True,
)