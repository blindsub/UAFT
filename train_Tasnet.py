import torch
from torch.utils.data import DataLoader as Loader
from data_loader.Dataset import Datasets, Datasets_tgt
import random
from config import option
import argparse
from logger import set_logger
import logging
from model import model
from utils.writer import MyWriter
import trainer_Tasnet

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dataloader(opt):
    # make validation dataloader

    val_dataset = Datasets(
        opt['datasets']['val']['dataroot_mix'],
        [opt['datasets']['val']['dataroot_targets'][0],
         opt['datasets']['val']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    val_dataloader = Loader(val_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            shuffle=False,
                            pin_memory=False)

    train_dataset = Datasets(
        opt['datasets']['train']['dataroot_mix'],
        [opt['datasets']['train']['dataroot_targets'][0],
         opt['datasets']['train']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    train_dataloader = Loader(train_dataset,
                              batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                              num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                              shuffle=opt['datasets']['dataloader_setting']['shuffle'],
                              pin_memory=False)



    tgt_train_dataset = Datasets_tgt(opt['datasets']['tgt_train']['dataroot_mix'],
                                     **opt['datasets']['audio_setting'])

    tgt_train_dataloader = Loader(tgt_train_dataset,
                                  batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                                  num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                                  shuffle=opt['datasets']['dataloader_setting']['shuffle'],
                                  pin_memory=False)

    tgt_val_dataset = Datasets(
        opt['datasets']['tgt_val']['dataroot_mix'],
        [opt['datasets']['tgt_val']['dataroot_targets'][0],
         opt['datasets']['tgt_val']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])

    tgt_val_dataloader = Loader(tgt_val_dataset,
                                batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                                num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                                shuffle=False,
                                pin_memory=False)

    return train_dataloader, val_dataloader, tgt_train_dataloader, tgt_val_dataloader

def make_encoder_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['e_lr'], weight_decay=opt['optim']['weight_decay'],betas=(0.5, 0.999))
    else:
        optimizer = optimizer(params, lr=opt['optim']['e_lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer

def make_domain_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['d_lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer

def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training Conv-TasNet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    init_random_seed(None)

    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build domian classifer
    logger.info("Building the domain classifer")
    domain_classifer = model.Domain_Classifer()
    domain_classifer.apply(inplace_relu)

    # build encoder
    logger.info("Building the encoder of both source domain")
    src_encoder = model.Encoder(kernel_size=16, out_channels=512)
    src_encoder.apply(inplace_relu)

    logger.info("Building the encoder of target domain")
    tgt_encoder = model.Encoder(kernel_size=16, out_channels=512)
    tgt_encoder.apply(inplace_relu)

    # build separation
    logger.info("Building the separation of source and target domain")
    src_separation = model.Separation_TasNet(repeats=3, conv1d_block=8, in_channels=512,
                                         out_channels=128, out_sp_channels=512, kernel_size=3,
                                         norm="gln", causal=False, num_spks=2)
    src_separation.apply(inplace_relu)

    logger.info("Building the separation of target domain")
    tgt_separation = model.Separation_TasNet(repeats=3, conv1d_block=8, in_channels=512,
                                             out_channels=128, out_sp_channels=512, kernel_size=3,
                                             norm="gln", causal=False, num_spks=2)
    tgt_separation.apply(inplace_relu)


    # build decoder
    logger.info("Building the decoder of both source and target domain")
    decoder = model.Decoder(in_channels=512, out_channels=1, kernel_size=16, stride=16 // 2)

    # build optimizer
    logger.info("Building the optimizer of all model")
    optimizer_tgt = make_encoder_optimizer([{'params':tgt_encoder.parameters()},
                                            {'params':tgt_separation.parameters()}] ,opt)

    optimizer_critic = make_domain_optimizer(domain_classifer.parameters(), opt)

    # build dataloader
    logger.info('Building the dataloader')
    train_dataloader, val_dataloader, tgt_train_dataloader, tgt_val_dataloader = make_dataloader(opt)

    logger.info('Train Datasets Length: {}, Val Datasets Length: {},'.format(
        len(train_dataloader), len(val_dataloader)))
    logger.info('Target Train Datasets Length: {}, Target Val Datasets Length: {},'.format(
        len(tgt_train_dataloader), len(tgt_val_dataloader)))


    log_dir = opt['logger']['log_dir']
    writer = MyWriter(log_dir)

    logger.info('Building the Trainer of Conv-Tasnet')
    trainer = trainer_Tasnet.Trainer(train_dataloader, val_dataloader, tgt_train_dataloader, tgt_val_dataloader,
                                     src_encoder, tgt_encoder, domain_classifer, src_separation, tgt_separation, decoder, optimizer_tgt,
                                     optimizer_critic, opt, writer)
    trainer.run()

if __name__ == "__main__":
    train()