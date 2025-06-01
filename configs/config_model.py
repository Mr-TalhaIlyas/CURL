

pretrained_root_dir = '/home/user01/Data/fetal/new_scripts/models/pretrained/'

mme = dict(
            num_classes = 2, # wiht sigmoid activation output
            
            slowfast_pretrained_chkpts= f"{pretrained_root_dir}slowfast_r50_4x16x1_kinetics400-rgb.pth",
            vidmae_pretrained_chkpts= f"{pretrained_root_dir}vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth",
            dropout_ratio=0.3,


            # num_persons=1,
            # backbone_in_channels=3,
            # head_in_channels=256,
            # body_pretrainned_chkpts= f"{pretrained_root_dir}gcn_body_17kpts_kinetic400.pth",
            # hand_pretrainned_chkpts= f"{pretrained_root_dir}gcn_hand_21kpts_fphad45.pth",
            # face_pretrainned_chkpts= None,

            # ewt_head_ch = 768,
            # mod_feats = 256,
            # ewt_dropout_ratio = 0.3,
            # ewt_pretrainned_chkpts=f'{pretrained_root_dir}ecg_vit_fold1flip.pth',

            # fusion_in_channels = 256,
            # fusion_heads = 3,
            # pose_fusion_dropout=0.3, 
            # mod_fusion_dropout=0.3,
            )
mae = dict(
            batch_size= 4,
            num_classes = 2, # wiht sigmoid activation output
            mask_ratio = 0.75,
            weight_decay = 0.05,
            lr = 0.001,
            blr = 1e-3,
            min_lr = 0.0,
            warmup_epochs = 10,#40,
            # mae_vit_large_patch16
            decoder_embed_dim = 512,
            decoder_depth = 8,
            decoder_num_heads = 16,
            t_patch_size = 2,
            num_frames = 16,
            sampling_rate = 4,
            bias_wd = False,
            clip_grad = 0.02,
            fp32 = True,

)

mae_finetune = dict(
            accum_iter = 64, # Eff.Batch.Size=Batch.Size × Accumulation Steps
            epochs = 30,#100
            repeat_aug = 2,
            batch_size = 4, 
            smoothing = 0.1,
            mixup =  0.8,
            cutmix =1.0,
            mixup_prob = 1.0,
            blr = 2e-4, #1e-4,#4.8e-3,#0.0024,
            num_frames = 16,
            sampling_rate = 4,
            dropout =  0.5, #0.3,
            warmup_epochs = 3,#10
            weight_decay = 0.05, # not in FIneTune.md
            layer_decay =  0.8,#0.75,
            drop_path_rate = 0.3, #0.2,
            clip_grad = 5.0,
            fp32 = True
)