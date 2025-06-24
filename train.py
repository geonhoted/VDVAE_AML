from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import imageio

def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    t0 = time.time()
    vae.zero_grad()

    stats = vae.forward(data_input, target)

    stats['elbo'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
            p2.data.mul_(H.ema_rate)
            p2.data.add_(p1.data * (1 - H.ema_rate))
    t1 = time.time()
    stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm)
    return stats


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        stats = ema_vae.forward(data_input, target)
    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_func, batch_size):
    for x in DataLoader(data, batch_size=batch_size):
        break
    orig_image = x[0]
    preprocessed = preprocess_func(x)[0]
    return orig_image, preprocessed


def train_loop(H, data_train, data_valid, preprocess_func, vae, ema_vae,
               optimizer, scheduler, starting_epoch, iterate, cur_eval_loss, logprint):
    train_sampler = DistributedSampler(data_train, num_replicas=H.mpi_size, rank=H.rank)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_func, H.num_images_visualize) # Removed H.dataset as it's not used
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=train_sampler):
            if H.max_iters > 0 and iterate >= H.max_iters:
                logprint(f"Reached max_iters={H.max_iters}, stopping training.")
                # iteration의 제일 마지막에는 항상 sample return
                if H.rank == 0:
                    write_images(H, ema_vae, viz_batch_original, viz_batch_processed,
                                f'{H.save_dir}/samples-final-{iterate}.png', logprint)
                return
            data_input, target = preprocess_func(x)
            training_stats = training_step(H, data_input, target, vae, ema_vae, optimizer, iterate)
            stats.append(training_stats)
            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original, viz_batch_processed, f'{H.save_dir}/samples-{iterate}.png', logprint)

            iterate += 1
            iters_since_starting += 1

            # 일정 시간마다 "latest" 체크포인트를 저장!
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_func)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)


def evaluate(H, ema_vae, data_valid, preprocess_func):
    stats_valid = []
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    for x in DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler):
        data_input, target = preprocess_func(x)
        stats_valid.append(eval_step(data_input, target, ema_vae))
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(n_batches=len(vals), filtered_elbo=np.mean(finites), **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


def write_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logprint):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        reconstruction = ema_vae.decoder.forward_manual_latents(mb, zs[:i], t=0.1)
        batches.append(reconstruction)
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        sample = ema_vae.decoder.forward_uncond(mb, t=t)
        batches.append(sample)
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3]).astype(np.uint8) # Explicitly cast to uint8
    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def run_test_eval(H, ema_vae, data_test, preprocess_func, logprint):
    print('evaluating')
    stats = evaluate(H, ema_vae, data_test, preprocess_func)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def main():
    # Encoder/Decoder/VAE 생성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(
        image_channels=image_channels,
        base_width=base_width,
        custom_width_str=custom_width_str,
        block_str=encblock_str,
        bottleneck_multiple=bottleneck_multiple
    )
    decoder = Decoder(
        width_map=encoder.widths,
        zdim=zdim,
        bottleneck_multiple=bottleneck_multiple,
        output_res=image_size,
        block_str=decblock_str,
        num_mixtures=num_mixtures,
        low_bit=False
    )

    H, logprint = set_up_hyperparams(s=[])
    H.device = device
    H, data_train, data_valid_or_test, preprocess_func = set_up_data(H)
    vae, ema_vae = load_vaes(encoder, decoder, image_size, logprint)

    # 저장된 checkpoint에서 학습을 다시 시작하는 경우
    # load vae parameter
    if H.restore_path is not None:
        model_ckpt = f"{H.restore_path}-model.th"
        ckpt_vae = torch.load(model_ckpt, map_location=H.device)
        vae.load_state_dict(ckpt_vae)
        vae = vae.to(device)
        # load ema_vae parameter
        if H.restore_ema_path is not None:
            ema_ckpt = f"{H.restore_ema_path}-model-ema.th"
            ckpt_ema = torch.load(ema_ckpt, map_location=H.device)
            ema_vae.load_state_dict(ckpt_ema)
            ema_vae = ema_vae.to(H.device)
        print(f">> Loaded model & EMA from {H.restore_path}.")
    else:
        vae = vae.to(device)
        ema_vae = ema_vae.to(device)

    # generate optimizer & load optimizer
    optimizer, scheduler, starting_epoch, iterate, cur_eval_loss = load_opt(H, vae, logprint)
    if H.restore_optimizer_path is not None:
        optimizer_ckpt = torch.load(H.restore_optimizer_path, map_location=H.device)
        optimizer.load_state_dict(optimizer_ckpt)
        print(f">> Loaded optimizer state from {H.restore_optimizer_path}")

    # 실제 evaluation & test loop
    if H.test_eval:
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_func, logprint)
    else:
        train_loop(H, data_train, data_valid_or_test, preprocess_func, vae, ema_vae,
                   optimizer, scheduler, starting_epoch, iterate, cur_eval_loss, logprint)

if __name__ == "__main__":
    main()
