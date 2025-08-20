import torch.nn.functional as F

loss_fn_cv = F.cross_entropy

def loss_fn_tm(Y_hat, Y):
    Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
    Y = Y.reshape(-1,)
    return F.cross_entropy(Y_hat, Y)

def loss_fn_mt(model, Y_hat, Y):
    """Compute masked loss, padded values set to zero"""
    Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
    Y = Y.reshape(-1,)
    l = F.cross_entropy(Y_hat, Y, reduction='none')
    mask = (Y != model.tgt_pad_idx).float()
    return (l * mask).sum() / mask.sum()