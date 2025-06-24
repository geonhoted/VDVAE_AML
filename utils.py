from mpi4py import MPI
import os
import json
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import AdamW
from collections import defaultdict
import argparse
import time
import numpy as np
import subprocess


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3)


class DmolNet(nn.Module):
    def __init__(self, width, num_mixtures, low_bit=False):
        super().__init__()
        self.width = width
        self.num_mixtures = num_mixtures
        self.low_bit = low_bit
        self.out_conv = nn.Conv2d(width, num_mixtures * 10, kernel_size=1, stride=1, padding=0)

    def nll(self, px_z, x):
        return discretized_mix_logistic_loss(x=x, l=self.forward(px_z), low_bit=self.low_bit)

    def forward(self, px_z):
        if not isinstance(px_z, torch.Tensor):
            if isinstance(px_z, np.ndarray):
                px_z = torch.from_numpy(px_z).to(device=self.out_conv.weight.device, dtype=self.out_conv.weight.dtype).contiguous()
        xhat = self.out_conv(px_z)
        return xhat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z), self.num_mixtures)
        xhat = (im + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def mpi_size():
    return MPI.COMM_WORLD.Get_size()


def mpi_rank():
    return MPI.COMM_WORLD.Get_rank()


def compute_mpi_topology():
    world_size = mpi_size()
    global_rank = mpi_rank()
    # compute num_nodes
    if world_size % 8 == 0:
        num_nodes = world_size // 8
    else:
        num_nodes = world_size // 8 + 1
    # compute gpus_per_nodes
    if world_size > 1:
        gpus_per_node = max(world_size // num_nodes, 1)
    else:
        gpus_per_node = 1
    # local rank
    local_rank = global_rank % gpus_per_node

    return world_size, local_rank, global_rank


def setup_mpi(H):
    H.mpi_size, H.local_rank, H.rank = compute_mpi_topology()
    os.environ["RANK"] = str(H.rank)
    os.environ["WORLD_SIZE"] = str(H.mpi_size)
    os.environ["MASTER_PORT"] = str(H.port)
    # os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["MASTER_ADDR"] = MPI.COMM_WORLD.bcast(socket.gethostname(), root=0)
    # 처음 한 번만 초기화
    if not dist.is_initialized():
      torch.cuda.set_device(H.local_rank)
      dist.init_process_group(backend='nccl', init_method="env://")   # remove f''


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')


def logger(log_prefix):
    'Prints the arguments out to stdout, .txt, and .jsonl files'
    jsonl_path = f'{log_prefix}.jsonl'
    txt_path = f'{log_prefix}.txt'

    def log(*args, pprint=False, **kwargs):
        if mpi_rank() != 0:
            return
        t = time.ctime()
        argdict = {'time': t}
        if len(args) > 0:
            argdict['message'] = ' '.join([str(x) for x in args])
        argdict.update(kwargs)

        txt_str = []
        args_iter = sorted(argdict) if pprint else argdict
        for k in args_iter:
            val = argdict[k]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            argdict[k] = val
            if isinstance(val, float):
                val = f'{val:.5f}'
            txt_str.append(f'{k}: {val}')
        txt_str = ', '.join(txt_str)

        if pprint:
            json_str = json.dumps(argdict, sort_keys=True)
            txt_str = json.dumps(argdict, sort_keys=True, indent=4)
        else:
            json_str = json.dumps(argdict)
        print(txt_str, flush=True)

        with open(txt_path, "a+") as f:
            print(txt_str, file=f, flush=True)
        with open(jsonl_path, "a+") as f:
            print(json_str, file=f, flush=True)
    return log


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_mpi(H)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    logprint('traning model', H.desc, 'on', H.dataset)
    return H, logprint


def linear_warmup(warmup_iters):
    def f(iteration):
        return iteration / warmup_iters if iteration < warmup_iters else 1.0
    return f


def load_vaes(encoder, decoder, image_size, logprint):
    mpi_size, local_rank, rank = compute_mpi_topology()
    torch.cuda.set_device(local_rank)

    vae = VAE(encoder, decoder, image_size).cuda(local_rank)
    ema_vae = VAE(encoder, decoder, image_size).cuda(local_rank)
    ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(True)

    if mpi_size > 1:
        vae = DistributedDataParallel(vae, device_ids=[local_rank], output_device=local_rank)
    # validate parameter names
    named = list(vae.named_parameters())
    all_params = list(vae.parameters())
    if len(named) != len(all_params):
        raise ValueError("Some parameters are unnamed-DDP requires all params to be named")
    total_params = 0
    for name, p in vae.named_parameters():
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae, ema_vae


def load_opt(H, vae, logprint):
    optimizer = AdamW(vae.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))

    starting_epoch = 0
    iterate = 0
    cur_eval_loss = float('inf')
    logprint('optimizer & scheduler initialized', epoch=starting_epoch, iterate=iterate, eval_loss=cur_eval_loss)
    return optimizer, scheduler, starting_epoch, iterate, cur_eval_loss


def allreduce(x, average):
    if mpi_size() > 1:
        dist.all_reduce(x, dist.ReduceOp.SUM)
    return x / mpi_size() if average else x


def get_cpu_stats_over_ranks(stat_dict):
    keys = sorted(stat_dict.keys())
    stats = torch.stack([torch.as_tensor(stat_dict[k]).detach().cuda().float() for k in keys])
    allreduced = allreduce(stats, average=True).cpu()
    return {k: allreduced[i].item() for (i, k) in enumerate(keys)}


def save_model(path, vae, ema_vae, optimizer, H):
    torch.save(vae.state_dict(), f'{path}-model.th')
    torch.save(ema_vae.state_dict(), f'{path}-model-ema.th')
    torch.save(optimizer.state_dict(), f'{path}-opt.th')
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])


def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['distortion_nans', 'rate_nans', 'skipped_updates', 'gcskip']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'elbo':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['elbo'] = np.mean(vals)
            z['elbo_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = stats[-1][k] if len(stats) < frequency else np.mean([a[k] for a in stats[-frequency:]])
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z
