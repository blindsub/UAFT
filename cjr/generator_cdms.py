import os
import glob
import tqdm
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.hparams import HParam

"""
python generator_cdms.py -c /27T/datasets/lunwen/conv-tasnet/code/convtasnet_adda_vctk__cdms/config/config.yaml -d /27T/datasets/lunwen/speechdatasets/convtasnet/rowstcmds -o /27T/datasets/lunwen/speechdatasets/convtasnet/cmds_sr8000
"""


def formatter(dir_, form, num):  # 替换
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):  # 语音活性检测，返回一条语音额全部大于20db片段
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(hp, args, num, s1_target, s2, train):  #
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')  # ！
    if train == 0:
        dir_ = os.path.join(args.out_dir, 'train')  # ！
    elif train == 1:
        dir_ = os.path.join(args.out_dir, 'val')
    elif train == 2:
        dir_ = os.path.join(args.out_dir, 'test')
    # print(s1_target)
    # print(s2)
    # raise EOFError

    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    if len(w1) < 100 or len(w2) < 100:
        return


    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    # if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
    #     return

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    # save vad & normalized wav files
    w1_path = formatter(dir_, hp.form.w1.wav, num)  # ！
    w2_path = formatter(dir_, hp.form.w2.wav, num)
    mixed_wav_path = formatter(dir_, hp.form.mixed.wav, num)   # ！
    librosa.output.write_wav(w1_path, w1, srate)
    librosa.output.write_wav(w2_path, w2, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude spectrograms
    # target_mag, _ = audio.wav2spec(w1)
    # mixed_mag, _ = audio.wav2spec(mixed)
    # target_mag_path = formatter(dir_, hp.form.target.mag, num)
    # mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)
    # torch.save(torch.from_numpy(target_mag), target_mag_path)
    # torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # save selected sample as text file. d-vec will be calculated soon
    # dvec_text_path = formatter(dir_, hp.form.dvec, num)  # ！
    # with open(dvec_text_path, 'w') as f:
    #     f.write(s1_dvec)

    w1_text_path = formatter(dir_, hp.form.w1.path, num)
    with open(w1_text_path, 'w') as f:
        f.write(s1_target)
    w2_text_path = formatter(dir_, hp.form.w2.path, num)
    with open(w2_text_path, 'w') as f:
        f.write(s2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--libri_dir', type=str, default=None,
                        help="Directory of LibriSpeech dataset, containing folders of train-clean-100, train-clean-360, train-other-500, dev-clean.")
    parser.add_argument('-v', '--voxceleb_dir', type=str, default=None,
                        help="Directory of VoxCeleb2 dataset, ends with 'aac'")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'val'), exist_ok=True)

    hp = HParam(args.config)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-100', '*'))
                            if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train-clean-360', '*'))
                            if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train-other-500', '*'))
                            if os.path.isdir(x)]

        val_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]
        test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'test', '*'))]
    # elif args.voxceleb_dir is not None:
    #     all_folders = [x for x in glob.glob(os.path.join(args.voxceleb_dir, '*'))
    #                         if os.path.isdir(x)]
    #     train_folders = all_folders[:-20]
    #     test_folders = all_folders[-20:]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    val_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                for spk in val_folders]
    val_spk = [x for x in val_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                    for spk in test_folders]
    test_spk = [x for x in test_spk if len(x) >= 2]


    def train_wrapper(num):
        spk1, spk2 = random.sample(train_spk, 2)
        # s1_dvec, s1_target = random.sample(spk1, 2)
        s1 = random.choice(spk1)
        s2 = random.choice(spk2)
        mix(hp, args,num, s1, s2, train=0)

    def val_wrapper(num):
        spk1, spk2 = random.sample(val_spk, 2)
        s1 = random.choice(spk1)
        s2 = random.choice(spk2)
        mix(hp, args, num, s1, s2, train=1)

    def test_wrapper(num):
        spk1, spk2 = random.sample(test_spk, 2)
        s1 = random.choice(spk1)
        s2 = random.choice(spk2)
        mix(hp, args, num, s1, s2, train=2)

    arr = list(range(int((10**6))))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))

    arr = list(range(int(2 * (10 ** 5))))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(val_wrapper, arr), total=len(arr)))

    arr = list(range(int(2*(10**5))))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))

    # for i in range(10**6):
    #     train_wrapper(i)
    #     if i %500 == 0:
    #         print("train:{}/{}".format(i,10**6))
    #
    # for i in range(10**3):
    #     train_wrapper(i)
    #     if i %100 == 0:
    #         print("test:{}/{}".format(i,10**3))