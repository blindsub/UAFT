import sys
sys.path.append('../')

import numpy as np

from utils.util import check_parameters
import time
import logging
from model.loss import Loss
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True



def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, tgt_train_dataloader, tgt_val_dataloader,
                 src_encoder, tgt_encoder, domain_classifer, src_separation, tgt_separation , decoder, optimizer_tgt,
                                     optimizer_critic, opt, writer):
        super(Trainer).__init__()
        self.writer = writer

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tgt_train_dataloader = tgt_train_dataloader
        self.tgt_val_dataloader = tgt_val_dataloader
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']
        self.batch_size = opt['datasets']['dataloader_setting']['batch_size']

        self.print_freq = opt['logger']['print_freq']
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.checkpoint_encoder = opt['train']['src_encoder']
        self.checkpoint_separation = opt['train']['separation']
        self.name = opt['name']

        if opt['train']['gpuid']:
            self.logger.info('Load Nvida GPU .....')
            self.device = torch.device(
                'cuda:{}'.format(opt['train']['gpuid'][0]))
            self.gpuid = opt['train']['gpuid']
            self.src_encoder = src_encoder.to(self.device)
            self.tgt_encoder = tgt_encoder.to(self.device)
            self.domian_classifer = domain_classifer.to(self.device)
            self.src_separation = src_separation.to(self.device)
            self.tgt_separation = tgt_separation.to(self.device)
            self.decoder = decoder.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.src_encoder)+
                                                                   check_parameters(self.tgt_encoder)+
                                                                   check_parameters(self.domian_classifer)+
                                                                   check_parameters(self.src_separation)+
                                                                   check_parameters(self.tgt_separation)+
                                                                   check_parameters(self.decoder)))

        if opt['resume']['state']:
            ckp = torch.load(opt['resume']['path'], map_location='cpu')
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))

            # restore src_encoder
            se_model_dict = self.src_encoder.state_dict()
            ss_model_dict = self.src_separation.state_dict()
            de_model_dict = self.decoder.state_dict()


            se_pretrained_dict = {k.replace('encoder.conv1d','conv1d'): v for k, v in ckp['model_state_dict'].items() if k.replace('encoder.conv1d','conv1d') in se_model_dict}
            se_model_dict.update(se_pretrained_dict)
            self.src_encoder.load_state_dict(se_model_dict)

            # restore src_separation
            ss_pretrained_dict = {k.replace('separation.',''): v for k, v in ckp['model_state_dict'].items() if k.replace('separation.','') in ss_model_dict}
            ss_model_dict.update(ss_pretrained_dict)
            self.src_separation.load_state_dict(ss_model_dict)


            # init weights of target encoder and separation with those of source encoder
            self.tgt_encoder.load_state_dict(self.src_encoder.state_dict())
            self.tgt_separation.load_state_dict(self.src_separation.state_dict())

            # restore decoder
            de_pretrained_dict = {k.replace('decoder.',''):v for k,v in ckp['model_state_dict'].items() if k.replace('decoder.','') in de_model_dict}
            de_model_dict.update(de_pretrained_dict)
            self.decoder.load_state_dict(de_model_dict)

            self.optimizer_tgt = optimizer_tgt
            self.optimizer_critic = optimizer_critic


        else:
            raise RuntimeError
            self.encoder = encoder.to(self.device)
            self.domian_classifer = domain_classifer.to(self.device)
            self.separation_network = separation.to(self.device)
            self.decoder = decoder.to(self.device)
            self.optimizer = optimizer

        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info(
                "Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0

    def train(self, epoch):
        self.logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.tgt_encoder.train()
        self.domian_classifer.train()
        self.tgt_separation.train()
        self.src_separation.eval()
        self.src_encoder.eval()

        criterion = nn.BCEWithLogitsLoss()
        num_index = 1
        for batch_id, ((mix, ref),(tgt_mix)) in enumerate(zip(self.train_dataloader, self.tgt_train_dataloader)):
            mix = mix.to(self.device)
            # ref = [ref[i].to(self.device) for i in range(self.num_spks)]
            tgt_mix = tgt_mix.to(self.device)
            self.optimizer_critic.zero_grad()

            try:
                if self.gpuid:
                    src_encoder = torch.nn.DataParallel(self.src_encoder)
                    tgt_encoder = torch.nn.DataParallel(self.tgt_encoder)
                    src_separation = torch.nn.DataParallel(self.src_separation)
                    tgt_separation = torch.nn.DataParallel(self.tgt_separation)
                    # separation = torch.nn.DataParallel(self.separation)
                    domain_classifer = torch.nn.DataParallel(self.domian_classifer)

                    ###########################
                    # 2.1 train discriminator #
                    ###########################
                    src_encoder_x = src_encoder(mix)
                    src_separation_x = src_separation(src_encoder_x)
                    src_separation_x = src_separation_x.permute(1, 0, 2, 3)
                    src_audio_encoder = [src_encoder_x * src_separation_x[i] for i in range(self.num_spks)]
                    src_mid_out = torch.cat(src_audio_encoder, dim=0)


                    tgt_encoder_x = tgt_encoder(tgt_mix)

                    tgt_separation_x = tgt_separation(tgt_encoder_x)
                    tgt_separation_x = tgt_separation_x.permute(1,0,2,3)
                    tgt_audio_encoder = [tgt_encoder_x * tgt_separation_x[i] for i in range(self.num_spks)]
                    tgt_mid_out = torch.cat(tgt_audio_encoder)
                    src_pred_concat = domain_classifer(src_mid_out.detach())
                    tgt_pred_concat = domain_classifer(tgt_mid_out.detach())
                    pred_concat = torch.cat((src_pred_concat, tgt_pred_concat), dim=0)


                    # reverse label
                    label_tgt = Variable(make_variable(torch.ones(2 * tgt_mix.size(0)).long()).type(torch.float32),
                                         requires_grad=False)
                    label_src = Variable(make_variable(torch.zeros(2 * mix.size(0)).long().type(torch.float32)),
                                         requires_grad=False)


                    label_concat = torch.cat((label_src, label_tgt), 0)

                    pred_concat = pred_concat.squeeze()

                    loss_critic = criterion(pred_concat, label_concat)

                    loss_critic.backward()

                    self.optimizer_critic.step()


                    ############################
                    # 2.2 train target encoder #
                    ############################

                    # zero gradients for optimizer
                    self.optimizer_critic.zero_grad()
                    self.optimizer_tgt.zero_grad()
                    tgt_encoder_y = tgt_encoder(tgt_mix)
                    tgt_separation_y = tgt_separation(tgt_encoder_y)
                    tgt_separation_y = tgt_separation_y.permute(1, 0, 2, 3)
                    tgt_audio_encoder_y = [tgt_encoder_y * tgt_separation_y[i] for i in range(self.num_spks)]
                    tgt_mid_out = torch.cat(tgt_audio_encoder_y,dim=0)
                    pred_tgt_y = domain_classifer(tgt_mid_out)
                    pred_tgt_y = pred_tgt_y.squeeze()

                    # prepare fake labels
                    label_tgt = Variable(make_variable(torch.zeros(2*tgt_mix.size(0))),requires_grad=False)

                    # compute loss for target encoder
                    loss_tgt = criterion(pred_tgt_y, label_tgt)
                    loss_tgt.backward()
                    # optimize target encoder
                    self.optimizer_tgt.step()



            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e


            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, e_lr:{:.3e}, d_lr:{:.3e}, loss_critic:{:.3f}, loss_tgt:{:.3f}'.format(
                    epoch, num_index, self.optimizer_tgt.param_groups[0]['lr'],self.optimizer_critic.param_groups[0]['lr'],
                    loss_critic.item(),loss_tgt.item())
                print('loss_tgt:', pred_tgt_y)
                print('loss_critic', pred_concat)
                self.logger.info(message)

            if num_index % 100 == 0:
                iter_loss = self.validation(self.cur_epoch)
                print('iter_loss',iter_loss)
                self.tgt_encoder.train()
                self.domian_classifer.train()
                self.tgt_separation.train()

            num_index += 1


        return 0

    def validation(self, epoch):
        t_sum_target = 0
        self.tgt_encoder.eval()
        self.tgt_separation.eval()
        self.src_separation.eval()
        self.src_encoder.eval()
        self.decoder.eval()

        num_batchs = len(self.tgt_val_dataloader)
        num_index = 1
        total_target_separation_loss = 0.0
        with torch.no_grad():
            for batch_id, (tgt_mix,tgt_ref) in enumerate(self.tgt_val_dataloader):

                tgt_mix = tgt_mix.to(self.device)
                tgt_ref = [tgt_ref[i].to(self.device) for i in range(self.num_spks)]
                self.optimizer_tgt.zero_grad()
                self.optimizer_critic.zero_grad()
                if self.gpuid:
                    tgt_encoder = torch.nn.DataParallel(self.tgt_encoder)
                    tgt_separation = torch.nn.DataParallel(self.tgt_separation)
                    decoder = torch.nn.DataParallel(self.decoder)



                tgt_encoder_x = tgt_encoder(tgt_mix)
                tgt_separation_x = tgt_separation(tgt_encoder_x)
                tgt_separation_x = tgt_separation_x.permute(1, 0, 2, 3)

                tgt_audio_encoder = [tgt_encoder_x * tgt_separation_x[i] for i in range(self.num_spks)]
                tgt_audio = [decoder(tgt_audio_encoder[i]) for i in range(self.num_spks)]

                tgt_l = Loss(tgt_audio, tgt_ref)
                target_separation_loss = tgt_l
                total_target_separation_loss += target_separation_loss.item()


                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, tgt_loss:{:.3f} >'.format(
                        epoch, num_index, total_target_separation_loss/self.print_freq)
                    self.logger.info(message)
                    t_sum_target += total_target_separation_loss
                    total_target_separation_loss = 0.0


                num_index += 1
        return t_sum_target/num_batchs

    def run(self):
        train_loss = []
        val_loss = []
        with torch.cuda.device(self.gpuid[0]):
            v_loss = self.validation(self.cur_epoch)
            # best_loss = v_loss+300
            best_loss = v_loss
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, v_loss))
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(self.cur_epoch)
                v_loss = self.validation(self.cur_epoch)


                train_loss.append(t_loss)
                val_loss.append(v_loss)


                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info(
                        'No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, best=True)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                        self.cur_epoch, best_loss))

                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_improve))
                    break
            self.save_checkpoint(self.cur_epoch, best=False)
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.total_epoch))

        # draw loss image
        plt.title("Loss of train and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, val_loss, 'c-', label=u'val_loss', linewidth=0.8)
        plt.legend()
        #plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')

    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'tgt_encoder_state_dict': self.tgt_encoder.state_dict(),
            'tgt_separation_state_dict':self.tgt_separation.state_dict(),
            'decoder':self.decoder.state_dict(),
            'domain_classifer_state_dict': self.domian_classifer.state_dict(),
            'optim_state_dict_d': self.optimizer_critic.state_dict(),
            'optim_state_dict_e': self.optimizer_tgt.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))
