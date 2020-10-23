import torch
from mir_eval.separation import bss_eval_sources
from itertools import permutations
import copy

def cal_sdr(ref, out, random_idx):
    num_spks = len(ref)  # 2
    ref_copy = copy.deepcopy(ref)
    out_copy = copy.deepcopy(out)

    for i in range(num_spks):
        out_copy[i] = out_copy[i].cpu().detach().numpy()
        ref_copy[i] = ref_copy[i].cpu().detach().numpy()

    for i in range(num_spks):
        ref_copy[i] = ref_copy[i][random_idx, :]
        out_copy[i] = out_copy[i][random_idx, :]

    def sdr(permute):
        return sum([bss_eval_sources(ref_copy[s], out_copy[t], False)[0][0]
                for s,t in enumerate(permute)]) / len(permute)


    sdr_mat = torch.stack(
        [torch.tensor(sdr(p)) for p in permutations(range(num_spks))])


    max_psdr, _ = torch.max(sdr_mat, dim=0)
    return max_psdr
