import numpy as np
from tensorboardX import SummaryWriter



class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_val(self, val_loss, step):

        self.add_scalar('test_loss', val_loss, step)
        
    def log_sdr(self, mean_sdr, epoch):
        self.add_scalar('src_SDR', mean_sdr, epoch)

    def log_tgt_sdr(self, mean_sdr, epoch):
        self.add_scalar('tgt_SDR', mean_sdr, epoch)

