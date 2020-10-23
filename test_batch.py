import os
import torch
from data_loader.AudioData import AudioReader, write_wav, read_wav
import argparse
from torch.nn.parallel import data_parallel
from model.model import Encoder,Decoder,Separation_TasNet
from logger.set_logger import setup_logger
import logging
from config.option import parse
import tqdm
from utils import util, calculate
import numpy as np
from data_loader import reader

class Separation():
	def __init__(self, mix_path, yaml_path, model, gpuid):
		super(Separation, self).__init__()
		self.index_dict = util.handle_scp(mix_path)
		self.keys = list(self.index_dict.keys())
		opt = parse(yaml_path)

		encoder = Encoder(kernel_size=16, out_channels=512)
		separation = Separation_TasNet(repeats=3, conv1d_block=8, in_channels=512,
                                         out_channels=128, out_sp_channels=512, kernel_size=3,
                                         norm="gln", causal=False, num_spks=2)
		decoder = Decoder(in_channels=512, out_channels=1, kernel_size=16, stride=16 // 2)

		dicts = torch.load(model, map_location='cpu')  
		encoder.load_state_dict(dicts['tgt_encoder_state_dict'])
		decoder.load_state_dict(dicts['decoder'])
		separation.load_state_dict(dicts['tgt_separation_state_dict'])
			
		setup_logger(opt['logger']['name'], opt['logger']['path'],  # 配置logger
		                    screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
		self.logger = logging.getLogger(opt['logger']['name'])  # 创建logger对象
		self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
		
		self.encoder = encoder.cuda()
		self.separation = separation.cuda()
		self.decoder = decoder.cuda()
		self.device=torch.device('cuda:{}'.format(
		    gpuid[0]) if len(gpuid) > 0 else 'cpu')
		self.gpuid=tuple(gpuid)
		self.num_spks = 2
		self.sr = 8000

	def inference(self, file_path):
		with torch.no_grad():
			for key in tqdm.tqdm(self.keys):
				wav = reader.read_wav(self.index_dict[key])
				egs = torch.from_numpy(wav)
				egs = egs.to(self.device)
				norm = torch.norm(egs,float('inf')) 
				if len(self.gpuid) != 0:
					if egs.dim() == 1:
						egs = torch.unsqueeze(egs, 0)
					ests_encoder = self.encoder(egs)
					ests_separation = self.separation(ests_encoder)
					ests_separation = ests_separation.permute(1, 0, 2, 3)
					audio_encoder = [ests_encoder * ests_separation[i] for i in range(self.num_spks)]
					ests = [self.decoder(audio_encoder[i]) for i in range(self.num_spks)]
					spks = [torch.squeeze(s.detach().cpu()) for s in ests]
				else:
					if egs.dim() == 1:
						egs = torch.unsqueeze(egs, 0)
					ests=self.net(egs)
					spks=[torch.squeeze(s.detach()) for s in ests]
				index=0
				for s in spks:
					s = s[:egs.shape[1]]  
					#norm
					s = s*norm/torch.max(torch.abs(s))
					# s = s.unsqueeze(0)
					s = torch.squeeze(s)  # 改   
					s = s.numpy()  # 改
					index += 1
					os.makedirs(file_path+'/spk'+str(index), exist_ok=True)
				
					filename=file_path+'/spk'+str(index)+'/'+ key
					reader.write_wav(filename, s, self.sr)
				self.logger.info("Compute over {:d} utterances".format(len(wav)))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-mix_scp', type=str, default='tt_mix.scp', help='Path to mix scp file.')
	parser.add_argument(
		'-yaml', type=str, default='config/Conv_Tasnet/train.yml', help='Path to yaml file.')
	parser.add_argument(
		'-model', type=str, default=r'/27T/datasets/lunwen/conv-tasnet/code/convtasnet_adda_vctk__cdms_moreval_dropout_hardlabel_reverselabel_1000/checkpoint/Conv_Tasnet/best.pt', help="Path to model file.")
	parser.add_argument(
		'-gpuid', type=str, default='0', help='Enter GPU id number')
	parser.add_argument(
		'-save_path', type=str, default='model_vctk2cdms_adda', help='save result path')
	parser.add_argument(
		'-clean_path_1', type=str, default=r'/27T/datasets/lunwen/speechdatasets/convtasnet/chinese/test_cmds/w1')
	parser.add_argument(
		'-clean_path_2', type=str, default=r'/27T/datasets/lunwen/speechdatasets/convtasnet/chinese/test_cmds/w2')
	args = parser.parse_args()
	gpuid=[int(i) for i in args.gpuid.split(',')]
	separation=Separation(args.mix_scp, args.yaml, args.model, gpuid)
	separation.inference(args.save_path)

	est_path_1 = os.path.join(args.save_path, 'spk1')
	est_path_2 = os.path.join(args.save_path, 'spk2')

	sr = 8000
	calculate.calculate(args.clean_path_1, est_path_1, est_path_2, sr)

	calculate.calculate(args.clean_path_2, est_path_1, est_path_2, sr)


if __name__ == "__main__":
	main()