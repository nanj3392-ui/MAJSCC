from net.decoder import *
from net.encoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn


class WITT(nn.Module):
    def __init__(self, args, config):
        super(WITT, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:    #"config"通常代表配置文件或配置对象，而"logger"则通常指日志记录器。
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)

        self.pass_channel = config.pass_channel  # 是否通过信道
        self.channel = Channel(args, config)
        self.multiple_snr = args.multiple_snr.split(",")   #1,4,7,10,13
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])

        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):    #将提取到的特征，通过信道传输。
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR = None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:   #没有设定信噪比
            SNR = choice(self.multiple_snr)
            chan_param = SNR      #信噪比就是通道参数
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, chan_param, self.model)       #图像 snr


        CBR = feature.numel() / 2 / input_image.numel()    #用于获取张量中的元素数量  CBR 表示了特征元素数量与输入图像元素数量的比值  /2 表示分成了实部和虚部两部分。
        # Feature pass channel
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()
