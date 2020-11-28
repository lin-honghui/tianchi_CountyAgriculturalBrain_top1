from torch import optim


def make_optimizer(cfg_optimizer,model):
    cfg_optimizer = cfg_optimizer.copy()
    optimizer_type = cfg_optimizer.pop('type')
    if hasattr(optim,optimizer_type):
        params = model.parameters()
        optimizer = getattr(optim,optimizer_type)(params,**cfg_optimizer)
        return optimizer
    else:
        raise KeyError("optimizer not found. Got {}".format(optimizer_type))