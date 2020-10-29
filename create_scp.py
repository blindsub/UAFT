import os

train_mix_scp = 'tr_mix.scp'
train_s1_scp = 'tr_s1.scp'
train_s2_scp = 'tr_s2.scp'

tgt_train_mix_scp = 'tgt_tr_mix.scp'

test_mix_scp = 'tt_mix.scp'
test_s1_scp = 'tt_s1.scp'
test_s2_scp = 'tt_s2.scp'

tgt_train_mix = ''  # path of target mixtures

train_mix = ''  # path of source mixtures
train_s1 = ''   # path of source spk1
train_s2 = ''   # path of source spk2

test_mix = ''   # path of source mixtures
test_s1 = ''    # path of source spk1
test_s2 = ''    # path of source spk2


tr_mix = open(train_mix_scp,'w')
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        tr_mix.write(file+" "+root+'/'+file)
        tr_mix.write('\n')


tr_s1 = open(train_s1_scp,'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file+" "+root+'/'+file)
        tr_s1.write('\n')


tr_s2 = open(train_s2_scp,'w')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file+" "+root+'/'+file)
        tr_s2.write('\n')


tt_mix = open(test_mix_scp,'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        tt_mix.write(file+" "+root+'/'+file)
        tt_mix.write('\n')


tt_s1 = open(test_s1_scp,'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file+" "+root+'/'+file)
        tt_s1.write('\n')


tt_s2 = open(test_s2_scp,'w')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file+" "+root+'/'+file)
        tt_s2.write('\n')


tgt_tr_mix = open(tgt_train_mix_scp,'w')
for root, dirs, files in os.walk(tgt_train_mix):
    files.sort()
    for file in files:
        tgt_tr_mix.write(file+" "+root+'/'+file)
        tgt_tr_mix.write('\n')


cv_mix_scp = 'cv_mix.scp'
cv_s1_scp = 'cv_s1.scp'
cv_s2_scp = 'cv_s2.scp'

tgt_cv_mix_scp = 'tgt_cv_mix.scp'
tgt_cv_s1_scp = 'tgt_cv_s1.scp'
tgt_cv_s2_scp = 'tgt_cv_s2.scp'

tgt_cv_mix = ''  # path of target validation mixture
tgt_cv_s1 = ''  # path of target validation spk1
tgt_cv_s2 = ''  # path of target validation spk2

cv_mix = ''  # path of source validation mixture
cv_s1 = ''  # path of target validation spk1
cv_s2 = ''  # path of target validation spk2



cv_mix_file = open(cv_mix_scp,'w')
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        cv_mix_file.write(file+" "+root+'/'+file)
        cv_mix_file.write('\n')


cv_s1_file = open(cv_s1_scp,'w')
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s1_file.write(file+" "+root+'/'+file)
        cv_s1_file.write('\n')


cv_s2_file = open(cv_s2_scp,'w')
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s2_file.write(file+" "+root+'/'+file)
        cv_s2_file.write('\n')

tgt_cv_mix_file = open(tgt_cv_mix_scp,'w')
for root, dirs, files in os.walk(tgt_cv_mix):
    files.sort()
    for file in files:
        tgt_cv_mix_file.write(file+" "+root+'/'+file)
        tgt_cv_mix_file.write('\n')

tgt_cv_s1_file = open(tgt_cv_s1_scp,'w')
for root, dirs, files in os.walk(tgt_cv_s1):
    files.sort()
    for file in files:
        tgt_cv_s1_file.write(file+" "+root+'/'+file)
        tgt_cv_s1_file.write('\n')

tgt_cv_s2_file = open(tgt_cv_s2_scp,'w')
for root, dirs, files in os.walk(tgt_cv_s2):
    files.sort()
    for file in files:
        tgt_cv_s2_file.write(file+" "+root+'/'+file)
        tgt_cv_s2_file.write('\n')