from mir_eval.separation import bss_eval_sources
import librosa
import torch
import os
import numpy as np
from pystoi import stoi
from pypesq import pesq

def calculate(clean_path, est_path_1, est_path_2, sr):


	clean_filenames = os.listdir(clean_path)
	spk1_filenames = os.listdir(est_path_1)
	spk2_filenames = os.listdir(est_path_2)

	clean_filenames.sort()
	spk1_filenames.sort()
	spk2_filenames.sort()

	clean_wav_path = list()
	spk1_wav_path = list()
	spk2_wav_path = list()

	for name in clean_filenames:
		path = clean_path + '/' + name
		clean_wav_path.append(path)

	for name in spk1_filenames:
		path = est_path_1 + '/' + name
		spk1_wav_path.append(path)

	for name in spk2_filenames:
		path = est_path_2 + '/' + name
		spk2_wav_path.append(path)

	zip1 = zip(clean_wav_path, spk1_wav_path)
	zip2 = zip(clean_wav_path, spk2_wav_path)

	sdr1 = list()
	sdr2 = list()
	sdr_final = list()

	stoi1 = list()
	stoi2 = list()
	stoi_final = list()

	pesq1 = list()
	pesq2 = list()
	pesq_final = list()

	print('start calculate')
	print('-'*100)


	for clean, est in zip1:

		clean_wav, _ = librosa.load(clean, sr=sr)
		est_wav, _ = librosa.load(est, sr=sr)

		sdr = bss_eval_sources(clean_wav, est_wav, False)[0][0]
		sdr1.append(sdr)

		stoi_1 = stoi(clean_wav, est_wav, sr, extended=False)
		stoi1.append(stoi_1)

		pesq_1 = pesq(clean_wav, est_wav, sr)
		pesq1.append(pesq_1)

	print('1/2 done')
	print('-'*100)

	for clean, est in zip2:
		clean_wav, _ = librosa.load(clean, sr=sr)
		est_wav, _ = librosa.load(est, sr=sr)

		sdr = bss_eval_sources(clean_wav, est_wav, False)[0][0]
		sdr2.append(sdr)

		stoi_2 = stoi(clean_wav, est_wav, sr, extended=False)
		stoi2.append(stoi_2)

		pesq_2 = pesq(clean_wav, est_wav, sr)
		pesq2.append(pesq_2)
	print('1/1 done')
	print('-'*100)

	print('len',len(sdr1))
	for i in range(len(sdr1)):
		if sdr1[i] > sdr2[i]:
			sdr_final.append(sdr1[i])
		else:
			sdr_final.append(sdr2[i])

		if stoi1[i] > stoi2[i]:
			stoi_final.append(stoi1[i])
		else:
			stoi_final.append(stoi2[i])

		if pesq1[i] > pesq2[i]:
			pesq_final.append(pesq1[i])
		else:
			pesq_final.append(pesq2[i])

	print(len(sdr_final))  # nums of wav
	sdr_mean = np.mean(sdr_final)
	stoi_mean = np.mean(stoi_final)
	pesq_mean = np.mean(pesq_final)

	print('mean sdr:%.3f' % sdr_mean)
	print('mean stoi:%.3f' % stoi_mean)
	print('mean pesq:%.3f' % pesq_mean)
