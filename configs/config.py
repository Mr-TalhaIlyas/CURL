from configs import *
from configs import config_model

root_dir = '/home/user01/Data/fetal'

config = dict(
                gpus_to_use = '0',
                DPI = 300,
                LOG_WANDB= False,
                BENCHMARK= False,
                DEBUG = False,
                USE_EMA_UPDATES = False,
                ema_momentum = 0.999,
                sanity_check = False,
                project_name= 'Fetal Kicks CLR',
                experiment_name= 'MAE_finetune_f1_v5-cycle-2xlr2',#'vidmae_1-24',#'new_loader_f2-2-no-pre',
                remove_noise = False,

                log_directory= f"{root_dir}/logs/",
                checkpoint_path= f"{root_dir}/chkpts/",

                pretrained_chkpts= f"{root_dir}/new_scripts/models/pretrained/",

                folds =  f"{root_dir}/data/folds/",
                vid_dir = f'{root_dir}/data/vids_filtered/', # vids_filtered  vids
                flow_dir = f'{root_dir}/data/flow/',
                lbl_dir = f'{root_dir}/data/lbl_w_time/',

                pin_memory=  True, # to speed up data transfer to GPU mostly True
                num_workers= 4,# 2,,6
                persistent_workers= True, # to limit RAM memory usage
                prefetch_factor = 2, # 2 is good default

                num_fold = 1,

                # training settings
                batch_size= 4, # it'll be doubled if we are using SimCLR Loader

                # learning rate
                learning_rate= 0.001,
                lr_schedule= 'cos',
                # num_repeats_per_epoch = 60,
                epochs= 100,
                warmup_epochs= 2,
                WEIGHT_DECAY= 0.0005,
                AUX_LOSS_Weights= 0.4,
                # MAE Contrastive Learning Settings
                mae_contrastive = dict(
                    img_size=224,
                    patch_size=16,
                    in_chans=3,
                    embed_dim=1024,
                    depth=24,
                    num_heads=16,
                    mlp_ratio=4.0,
                    num_frames=16,
                    t_patch_size=2,
                    projection_dim=256,        # For spatial contrastive
                    temporal_projection_dim=128, # For temporal contrastive
                    use_cls_token=True,
                ),
                
                # Dual Contrastive Loss Settings (updated)
                enable_temporal_loss = True,
                spatial_loss_weight = 1.0,
                temporal_loss_weight = 0.5,
                temperature_spatial = 0.5,
                temperature_temporal = 0.1,
                tc_clusters = 8,
                tc_num_iters = 10,
                tc_do_entro = True,
                dual_loss_mode = 'both',  # 'spatial_only', 'temporal_only', 'both'

                # Temporal Contrastive Loss Parameters  
                tc_clusters = 10,                # Number of clusters for KMeans
                tc_num_iters = 10,             # KMeans iterations
                tc_do_entro = True,            # Enable IID regularization
                
                # Training with dual loss
                dual_loss_mode = 'both',       # 'spatial_only', 'temporal_only', 'both'
                
                # '''
                # Dataset
                # '''
                group_labels = False, # set to False for trainig
                video_fps = 23, # FPS
                sample_duration = 3, # from [3,5,7,10]
                y_seconds=30, # 30 set to 2 for evaluations
                noise_threshold=0,
                crop_minutes=2,
                chk_segments=False,
                downsmaple_clip=True,
                downsampled_frame_rate = 16, # 48 per "sample_duration" seconds

                video_height= 224,
                video_width= 224,

                num_classes = 2,
                classes = ['noise', 'non-movement', 'movement'],  # noise
                # super_classes = ['baseline', 'seizure'],
                
                # Model
                model = config_model.mme,
                mae = config_model.mae,
                mae_finetune = config_model.mae_finetune,
                # Fine-tuning specific parameters
                finetune_model_type = 'contrastive_mae',  # 'standard_mae' or 'contrastive_mae'
                contrastive_checkpoint_path = '/home/user01/Data/fetal/chkpts/contrastive_mae_best.pth',
                mae_checkpoint_path = '/home/user01/Data/fetal/chkpts/MAE_ViT_2.pth',
                
                # Training configuration
                loss_type = 'focal',  # 'focal', 'cross_entropy', 'label_smoothing'
                scheduler_type = 'cyclic',  # 'cyclic' or 'cosine'


                lbl_ints = {
                            "Noise": 0,
                            "None-Movement": 1,
                            "Head Motion": 2,
                            "Twitch": 3, # <- not in data
                            "Startle": 4,
                            "Wave": 5,
                            "Kick": 6,
                            "Pant": 7,
                            "Hiccups": 8,
                            "Trunk": 9,
                            "Precthl": 10,
                            "Limb": 11
                        },
                # head and twitch startle in one class
                grouped_labels = {
                    0: 0,  # Noise -> Removed in Data Loader
                    1: 1,  # None-Movement -> No Movements
                    2: 6,  # Head Motion -> Head and Facial Movements
                    3: 2,  # Twitch -> Small and Quick Movements
                    4: 2,  # Startle -> Small and Quick Movements
                    5: 3,  # Wave -> Limb Movements
                    6: 3,  # Kick -> Limb Movements
                    7: 4,  # Pant -> Respiratory Movements
                    8: 4,  # Hiccups -> Respiratory Movements
                    9: 5,  # Trunk -> Trunk and Large Body Movements
                    10: 5, # Precthl -> Trunk and Large Body Movements
                    11: 3  # Limb -> Limb Movements
                                },
                # noise (0th index) not included
                grouped_weights = [ 0.27572671, 39.45401896,  9.68617219,  1.2504065 ,  2.24772325],

                class_weights = {0: 3.452722365340438, 1: 0.5046390381286912, 2: 1.3721956727741944},

                class_weight_all = {'0': 0.8631805913351095, '1': 0.1261597595321728,
                                    '2': 105.52644122925078, '3': 23.630544253566406,
                                    '4': 307.17393078681033, '5': 39.825343915343915,
                                    '6': 15.443400794432408, '7': 0.639876489306644,
                                    '8': 10.584697378782765, '9': 3.7166797024677165,
                                    '10': 1.4030222709344546, '11': 7.63113979593492}
                # head and twitch seprate
                # grouped_labels7 = {
                #                     0: 0,  # Noise -> General Movements
                #                     1: 1,  # None-Movement -> General Movements
                #                     2: 2,  # Head Motion -> Head and Facial Movements
                #                     3: 3,  # Twitch -> Small and Quick Movements
                #                     4: 3,  # Startle -> Small and Quick Movements
                #                     5: 4,  # Wave -> Limb Movements
                #                     6: 4,  # Kick -> Limb Movements
                #                     7: 5,  # Pant -> Respiratory Movements
                #                     8: 5,  # Hiccups -> Respiratory Movements
                #                     9: 6,  # Trunk -> Trunk and Large Body Movements
                #                     10: 6, # Precthl -> Trunk and Large Body Movements
                #                     11: 4  # Limb -> Limb Movements
                #                 },
                )

# [0.1261597595321728,
# 105.52644122925078, 23.630544253566406,
# 307.17393078681033,39.825343915343915,
# 15.443400794432408,0.639876489306644,
# 10.584697378782765,3.7166797024677165,
# 1.4030222709344546, 7.63113979593492]