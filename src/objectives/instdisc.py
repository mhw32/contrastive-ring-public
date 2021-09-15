import torch
import torch.nn.functional as F
from src.utils.utils import l2_normalize


class NCE(object):

    def __init__(
            self,
            indices,
            outputs,
            memory_bank,
            k=4096,
            t=0.07,
            m=0,
            **kwargs
        ):
        super().__init__()
        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)
        self.memory_bank = memory_bank
        self.device = outputs.device
        self.data_len = memory_bank.size
        self.k, self.t, self.m = k, t, m
    
    def updated_new_data_memory(self):
        data_memory = self.memory_bank.at_idxs(self.indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * self.outputs
        return l2_normalize(new_data_memory, dim=1)

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_pos = self.memory_bank.get_dot_products(self.outputs, self.indices)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k - 1), device=self.device).long()
        noise_indx = torch.cat([self.indices.unsqueeze(1), noise_indx], dim=1)
        witness_partition = self.memory_bank.get_dot_products(self.outputs, noise_indx)
        witness_partition = torch.logsumexp(witness_partition / self.t, dim=1)
        loss = -torch.mean(witness_pos / self.t - witness_partition)
        return loss


class NCEBall(NCE):

    def __init__(
            self,
            indices,
            outputs,
            memory_bank,
            k=4096,
            t=0.07,
            m=0,
            thres_outer=0.1,
            thres_inner=0.1,
            dynamic_inner=True,
            neighbor_views=False,
            logsumexp_inner=True,
            self_negative=True,
            **kwargs
        ):
        super().__init__(indices, outputs, memory_bank, k=k, t=t, m=m)
        if not dynamic_inner: assert thres_inner < thres_outer
        self.thres_outer = thres_outer
        self.thres_inner = thres_inner
        self.dynamic_inner = dynamic_inner
        self.neighbor_views = neighbor_views
        self.logsumexp_inner = logsumexp_inner
        self.self_negative = self_negative

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_pos = self.memory_bank.get_dot_products(
            self.outputs, 
            self.indices,
        ) / self.t

        all_dps = self.memory_bank.get_all_dot_products(self.outputs) / self.t
        bank_size = all_dps.size(0)

        outer_dps, _ = torch.topk(
            all_dps, 
            int(self.thres_outer * bank_size), 
            sorted=False,
            dim=1,
        )

        k = (self.k - 1) if self.self_negative else self.k
        noise_indx = torch.randint(
            0, outer_dps.size(1), (batch_size, k), device=self.device).long()
        outer_dps  = torch.gather(outer_dps, 1, noise_indx)
        
        if self.self_negative:
            outer_dps = torch.cat((witness_pos.unsqueeze(1), outer_dps), dim=1)

        outer_size = outer_dps.size(1)

        if self.neighbor_views:
            if self.dynamic_inner:
                # set the inner threshold as a fraction of the outer, and 
                # only set a policy for changing the outer
                inner_idx = int(self.thres_inner * outer_size)
                inner_dps, _ = torch.topk(outer_dps, inner_idx, sorted=False, dim=1)
            else:
                inner_idx = int(self.thres_inner * self.data_len)
                inner_dps, _ = torch.topk(all_dps, inner_idx, sorted=False, dim=1)

            witness_pos = torch.cat((witness_pos.unsqueeze(1), inner_dps), dim=1)
            if self.logsumexp_inner:  # add the positive view as a negative example
                witness_pos = torch.logsumexp(witness_pos, dim=1)
            else:
                witness_pos = torch.mean(witness_pos, dim=1)

        witness_partition = torch.logsumexp(outer_dps, dim=1)
        loss = -torch.mean(witness_pos - witness_partition)
        return loss


class NCERing(NCEBall):

    def get_loss(self):
        batch_size = self.outputs.size(0)
        witness_pos = self.memory_bank.get_dot_products(
            self.outputs, 
            self.indices,
        ) / self.t

        all_dps = self.memory_bank.get_all_dot_products(self.outputs) / self.t
        bank_size = all_dps.size(1)
        k = int(self.thres_outer * bank_size)
        sorted_dps, _ = torch.topk(all_dps, int(self.thres_outer * bank_size), dim=1)
        # sorted_dps, _ = torch.sort(all_dps, dim=1, descending=True)  <-- too slow

        if self.dynamic_inner:
            outer_idx = int(self.thres_outer * bank_size)
            outer_dps = sorted_dps[:, :outer_idx]
            inner_idx = int(self.thres_inner * outer_dps.size(1))
            ring_dps = outer_dps[:, inner_idx:]
        else:
            outer_idx = int(self.thres_outer * bank_size)
            inner_idx = int(self.thres_inner * bank_size)
            ring_dps = sorted_dps[:, inner_idx:outer_idx]
        
        k = (self.k - 1) if self.self_negative else self.k
        noise_indx = torch.randint(0, ring_dps.size(1), (batch_size, k), device=self.device).long()
        ring_dps  = torch.gather(ring_dps, 1, noise_indx)

        if self.self_negative:
            ring_dps = torch.cat((witness_pos.unsqueeze(1), ring_dps), dim=1)

        if self.neighbor_views:
            inner_dps = sorted_dps[:, :inner_idx]
            witness_pos = torch.cat((witness_pos.unsqueeze(1), inner_dps), dim=1)
            if self.logsumexp_inner:
                witness_pos = torch.logsumexp(witness_pos, dim=1)
            else:
                witness_pos = torch.mean(witness_pos, dim=1)

        witness_partition = torch.logsumexp(ring_dps, dim=1)
        loss = -torch.mean(witness_pos - witness_partition)
        return loss
