import numpy as np


def epoch_batches(n, batch_size, seed, epoch, shuffle=True, drop_last=False):
    """한 epoch의 batch 인덱스 목록. 섞는 순서는 (seed, epoch)로만 정해진다."""
    order = np.random.default_rng([seed, epoch]).permutation(n) if shuffle else np.arange(n)
    batches = [order[i:i + batch_size].tolist() for i in range(0, n, batch_size)]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    return batches


def assign_to_workers(batches, num_workers):
    """worker k가 batch k, k+W, k+2W, ...를 맡는다. 반환: worker별 batch 목록의 list."""
    return [batches[k::num_workers] for k in range(num_workers)]
