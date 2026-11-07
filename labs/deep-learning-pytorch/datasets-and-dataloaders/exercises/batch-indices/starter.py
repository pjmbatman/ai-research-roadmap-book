import numpy as np


def epoch_batches(n, batch_size, seed, epoch, shuffle=True, drop_last=False):
    """한 epoch의 batch 인덱스 목록. 섞는 순서는 (seed, epoch)로만 정해진다."""
    raise NotImplementedError("순서를 정하고 batch_size로 자르세요")


def assign_to_workers(batches, num_workers):
    """worker k가 batch k, k+W, k+2W, ...를 맡는다. 반환: worker별 batch 목록의 list."""
    raise NotImplementedError("batch를 차례로 돌아가며 나누세요")
