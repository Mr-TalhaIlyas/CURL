from termcolor import cprint
import torch, os, json
from configs.config import config
from timm.models.layers import trunc_normal_

def save_chkpt(model, optimizer, epoch=0, loss=0, acc=0, return_chkpt=False):
    cprint('-> Saving checkpoint', 'green')
    torch.save({
                'epoch': epoch,
                'loss': loss,
                'acc': acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}_epoch{epoch}.pth'))#_epoch{epoch}
    cprint(os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}_epoch{epoch}.pth'), 'cyan')#_epoch{epoch}
    if return_chkpt:
        return os.path.join(config["checkpoint_path"], f'{config["experiment_name"]}_epoch{epoch}.pth')#_epoch{epoch}

def load_chkpt(model, optimizer, chkpt_path):
    if os.path.isfile(chkpt_path):
        print(f'-> Loading checkpoint from {chkpt_path}')
        checkpoint = torch.load(chkpt_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']

        print(f'-> Loaded checkpoint for epoch {epoch} with loss {loss} and accuracy {acc}')
        return epoch, loss, acc
    else:
        print(f'Error: No checkpoint found at {chkpt_path}')
        return None, None, None
    
def load_pretrained_chkpt(model, pretrained_path=None):
        if pretrained_path is not None:
            chkpt = torch.load(pretrained_path,
                               map_location='cpu')
            try:
                # load pretrained
                del chkpt['state_dict']['backbone.A'] # delete the saved adjacency matrix 
                pretrained_dict = chkpt['state_dict']
                # load model state dict
                state = model.state_dict()
                # loop over both dicts and make a new dict where name and the shape of new state match
                # with the pretrained state dict.
                matched, unmatched = [], []
                new_dict = {}
                for i, j in zip(pretrained_dict.items(), state.items()):
                    pk, pv = i # pretrained state dictionary
                    nk, nv = j # new state dictionary
                    # if name and weight shape are same
                    if pk.strip('backbone.') == nk: #.strip('backbone.')
                        new_dict[nk] = pv
                        matched.append(pk)
                    elif pv.shape == nv.shape:
                        new_dict[nk] = pv
                        matched.append(pk)
                    else:
                        unmatched.append(pk)

                state.update(new_dict)
                model.load_state_dict(state)
                print('Pre-trained state loaded successfully...')
                print(f'Mathed kyes: {len(matched)}, Unmatched Keys: {len(unmatched)}')
                # print(unmatched)
            except:
                print(f'ERROR in pretrained_dict @ {pretrained_path}')
        else:
            print('Enter pretrained_dict path.')

def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed
            
def load_pretrained_MAE_chkpt(model, pretrained_path=None, interpolate_embed=True):
    if pretrained_path is not None:
        # Load the pretrained checkpoint
        chkpt = torch.load(pretrained_path, map_location='cpu')
        try:
            # Extract the pretrained state dictionary
            pretrained_dict = chkpt['model_state_dict']

            print(60*'%')
            for k in ["head.weight", "head.bias", "mask_token", "decoder_pos_embed"]:
                if (
                    k in pretrained_dict
                    # and chkpt[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_dict[k]
                    
            # Get the current model's state dictionary
            model_state = model.state_dict()
            
            if interpolate_embed:
                print("Interpolate position embeddings...")
                interpolate_pos_embed(model, model_state)
                print("Interpolation done.")
            if not interpolate_embed:
                print("Not interpolating position embeddings...")
            # Initialize lists to track matched and unmatched keys
            matched, unmatched = [], []

            # Find the keys in both dictionaries and create lists for any extra layers
            pretrained_keys = set(pretrained_dict.keys())
            model_keys = set(model_state.keys())

            # Layers present in the pretrained checkpoint but not in the model
            extra_pretrained_layers = pretrained_keys - model_keys
            # Layers present in the model but not in the pretrained checkpoint
            extra_model_layers = model_keys - pretrained_keys

            # Create a new state dictionary to be loaded into the model
            new_dict = {}

            # Loop over the model state and match with the pretrained state dict
            for key, value in model_state.items():
                if key in pretrained_dict and pretrained_dict[key].shape == value.shape:
                    # If the key exists in the pretrained state and has the same shape, use the pretrained weights
                    new_dict[key] = pretrained_dict[key]
                    matched.append(key)
                else:
                    unmatched.append(key)

            # Update the model state dictionary with the new matched state
            model_state.update(new_dict)
            model.load_state_dict(model_state)
            
            # Print out the status
            print('Pre-trained state loaded successfully...')
            print(f'Matched Keys: {len(matched)}, Unmatched Keys: {len(unmatched)}')
            if extra_pretrained_layers:
                print(f'Extra layers in pretrained model not found in new model: {len(extra_pretrained_layers)}')
            if extra_model_layers:
                print(f'Extra layers in new model not found in pretrained model: {len(extra_model_layers)}')

            print('Initializing model Head Weights...')
            _ = trunc_normal_(model.head.weight, std=2e-5)

            print(60*'%')
            
            return matched, unmatched, extra_pretrained_layers, extra_model_layers

        except Exception as e:
            print(f'ERROR loading pretrained_dict from {pretrained_path}: {e}')
    else:
        print('Enter pretrained_dict path.')
        return None

def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, fp32=False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in [
        "cls_token",
        "mask_token",
    ]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("pos_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers