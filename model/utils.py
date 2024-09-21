import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
        
        if args.model == 'vss':
            from .dim2 import VSS_Net
            return VSS_Net()
    
   
        if args.model == 'unet':
            from .dim2 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        if args.model == 'unet++':
            from .dim2 import UNetPlusPlus
            if pretrain:
                raise ValueError('No pretrain model available')
            
        if args.model == 'attention_unet':         ########2d attention_unet
            from .dim2 import AttentionUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return AttentionUNet(args.in_chan, args.classes, args.base_chan)
        
        # if args.model == 'attention_unet':
        #     from .dim3 import AttentionUNet
        #     if pretrain:
        #         raise ValueError('No pretrain model available')
        #     return AttentionUNet(args.in_chan, args.classes, args.base_chan)
        if args.model == 'utnetv2':
            from .dim2.utnetv2 import UTNetV2
            if pretrain:
                raise ValueError('No pretrain model available')
            return UTNetV2(args.in_chan, args.classes, args.base_chan)
        
        if args.model == 'SVSNet':
            from .dim2.svsnet import SVSNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return SVSNet(1,4,512,512)

        elif args.model == 'resunet':
            from .dim2 import UNet 
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        elif args.model == 'daunet':
            from .dim2 import DAUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return DAUNet(args.in_chan, args.classes, args.base_chan, block=args.block)

        elif args.model in ['medformer']:
            from .dim2 import MedFormer
            if pretrain:
                raise ValueError('No pretrain model available')
            return MedFormer(args.in_chan, args.classes, args.base_chan, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=nn.ReLU, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, aux_loss=args.aux_loss)


        elif args.model == 'transunet':
            from .dim2 import VisionTransformer as ViT_seg
            from .dim2.transunet import CONFIGS as CONFIGS_ViT_seg
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = args.classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(args.training_size[0]/16), int(args.training_size[1]/16))
            net = ViT_seg(config_vit, img_size=args.training_size[0], num_classes=args.classes)

            if pretrain:
                net.load_from(weights=np.load(args.init_model))

            return net
        
        elif args.model == 'swinunet':
            from .dim2 import SwinUnet
            from .dim2.swin_unet import SwinUnet_config
            config = SwinUnet_config()
            net = SwinUnet(config, img_size=512, num_classes=args.classes)
            
            if pretrain:
                net.load_from(args.init_model)

            return net

        elif args.model == 'multi_DEGCN':
            from .dim2.multi_DEGCN import DEDCGCNEE
            net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

            if pretrain:
                net.load_from(args.init_model)

            return net

        elif args.model == 'DE_DCGCN_EE':
            from .dim2.DE_DCGCN_EE import DEDCGCNEE
            net = DEDCGCNEE(args.in_chan, args.classes)

            if pretrain:
                net.load_from(args.init_model)
            
            return net

        elif args.model == 'Angionet':
            from .dim2.Angionet import AngioNet # type: ignore
            net = AngioNet()

            if pretrain:
                net.load_from(args.init_model)

            return net
                
        elif args.model == 'source_DE_DCGCN_EE':
            from .dim2.source_DE_DCGCN_EE import DEDCGCNEE
            net = DEDCGCNEE(args.in_chan, args.classes)

            if pretrain:
                net.load_from(args.init_model)



            return net

        elif args.model == 'res_unet':
            from .dim2.res_unet import res_unet
            net = res_unet(args.in_chan, args.classes)

            if pretrain:
                net.load_from(args.init_model)

            return net

        elif args.model == 'tau_gcn':
            from .dim2.tau_dcn import DEDCGCNEE
            # from .dim2.swin_unet import SwinUnet_config
            # config = SwinUnet_config()
            net = DEDCGCNEE(args.in_chan, args.classes)

            if pretrain:
                net.load_from(args.init_model)

            return net
            
        elif args.model == 'DE_DC_EE':

          from .dim2.DE_DC_EE import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes)

          if pretrain:
              net.load_from(args.init_model)

          return net
          
        elif args.model == 'multi_DEGCN_allmulti':

          from .dim2.multi_DEGCN_allmulti import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

          if pretrain:
              net.load_from(args.init_model)

          return net
          
        elif args.model == 'multi_DEGCN_multi':

          from .dim2.multi_DEGCN_multi import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

          if pretrain:
              net.load_from(args.init_model)

          return net

        elif args.model == 'multi_DEGCN_premulti':

          from .dim2.multi_DEGCN_premulti import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

          if pretrain:
              net.load_from(args.init_model)

          return net

        elif args.model == 'last':

          from .dim2.last import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

          if pretrain:
              net.load_from(args.init_model)

          return net

        elif args.model == 'nobd':

          from .dim2.nobd import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

          if pretrain:
              net.load_from(args.init_model)

          return net

        elif args.model == 'noatten':

          from .dim2.noatten import DEDCGCNEE

          net = DEDCGCNEE(args.in_chan, args.classes,args.frame_N)

          if pretrain:
              net.load_from(args.init_model)

          return net



   # elif args.dimension == '3d':
        elif args.model == 'vnet':
            from .dim3 import VNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return VNet(args.in_chan, args.classes, scale=args.downsample_scale, baseChans=args.base_chan)
        elif args.model == 'resunet':
            from .dim3 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'unet':
            from .dim3 import UNet
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'unet++':
            from .dim3 import UNetPlusPlus
            return UNetPlusPlus(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'attention_unet':
            from .dim3 import AttentionUNet
            return AttentionUNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'medformer':
            from .dim3 import MedFormer

            return MedFormer(args.in_chan, args.classes, args.base_chan, map_size=args.map_size, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, proj_type=args.proj_type, norm=args.norm, act=args.act, kernel_size=args.kernel_size, scale=args.down_scale, aux_loss=args.aux_loss)
    
        elif args.model == 'unetr':
            from .dim3 import UNETR
            model = UNETR(args.in_chan, args.classes, args.training_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed='perceptron', norm_name='instance', res_block=True)
            
            return model
        elif args.model == 'vtunet':
            from .dim3 import VTUNet
            model = VTUNet(args, args.classes)

            if pretrain:
                model.load_from(args)
            return model
        elif args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            model = SwinUNETR(args.window_size, args.in_chan, args.classes, feature_size=args.base_chan)

        elif args.model == 'utnetv2':
            from .dim3 import UTNetV2
            model = UTNetV2(in_chan=1, num_classes=2)

            if args.pretrain:
                weights = torch.load('/research/cbim/vast/yg397/ConvFormer/ConvFormer/initmodel/model_swinvit.pt')
                model.load_from(weights=weights)

            return model
        elif args.model == 'nnformer':
            from .dim3 import nnFormer
            model = nnFormer(args.window_size, input_channels=args.in_chan, num_classes=args.classes, deep_supervision=args.aux_loss)

            return model
    #else:
       # raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')

