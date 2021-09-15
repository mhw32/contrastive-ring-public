import torch
import torch.nn.functional as F
from src.utils.utils import l2_normalize


class SimCLR(object):

    def __init__(self, outputs1, outputs2, t=0.07, **kwargs):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t

    def get_loss(self):
        witness_pos = torch.sum(self.outputs1 * self.outputs2, dim=1)
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        witness_partition = self.outputs1 @ outputs12.T
        witness_partition = torch.logsumexp(witness_partition / self.t, dim=1)
        loss = -torch.mean(witness_pos / self.t - witness_partition)
        return loss


class SimCLRBall(SimCLR):

    def __init__(
            self,
            outputs1,
            outputs2,
            t=0.07,
            thres_outer=0.1,
            thres_inner=0.1,
            dynamic_inner=True,
            neighbor_views=False,
            logsumexp_inner=True,
            self_negative=True,
            **kwargs
        ):
        super().__init__(outputs1, outputs2, t=t)
        if not dynamic_inner: assert thres_inner < thres_outer
        self.thres_outer = thres_outer
        self.thres_inner = thres_inner
        self.dynamic_inner = dynamic_inner
        self.neighbor_views = neighbor_views
        self.logsumexp_inner = logsumexp_inner
        self.self_negative = self_negative

    def get_loss(self):
        witness_pos = torch.sum(
            self.outputs1 * self.outputs2, 
            dim=1,
        ) / self.t

        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        all_dps = self.outputs1 @ outputs12.T / self.t
        all_size = all_dps.size(1)

        # batch_size x elements inside outer ball
        outer_dps, _ = torch.topk(
            all_dps,
            int(self.thres_outer * self.k), 
            sorted=False, 
            dim=1,
        )

        if self.self_negative:
            # important to add self product to the outer dps
            outer_dps = torch.cat((witness_pos.unsqueeze(1), outer_dps), dim=1)

        outer_size = outer_dps.size(1)

        if self.neighbor_views:  # Ball does not need these views otherwise
            if self.dynamic_inner:  # choose from outer_dps set or all_dps set
                inner_idx = int(self.thres_inner * outer_size)
                inner_dps, _ = torch.topk(outer_dps, inner_idx, sorted=False, dim=1)
            else:
                inner_idx = int(self.thres_inner * all_size)
                inner_dps, _ = torch.topk(all_dps, inner_idx, sorted=False, dim=1)

            # optimize the self product and contents of inner ball
            witness_pos = torch.cat((witness_pos.unsqueeze(1), inner_dps), dim=1)
            if self.logsumexp_inner:
                witness_pos = torch.logsumexp(witness_pos, dim=1)
            else:
                witness_pos = torch.mean(witness_pos, dim=1)

        # compute the partition estimate
        witness_partition = torch.logsumexp(outer_dps, dim=1)
        loss = -torch.mean(witness_pos - witness_partition)
        return loss


class SimCLRRing(SimCLRBall):

    def get_loss(self):
        # compute the self product (scaled by temperature)
        witness_pos = torch.sum(
            self.outputs1 * self.outputs2, 
            dim=1, 
        ) / self.t

        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        all_dps = self.outputs1 @ outputs12.T / self.t
        sorted_dps, _ = torch.sort(all_dps, dim=1, descending=True)
        all_size = all_dps.size(1)

        if self.dynamic_inner:
            outer_idx = int(self.thres_outer * all_size)
            outer_dps = sorted_dps[:, :outer_idx]
            inner_idx = int(self.thres_inner * outer_dps.size(1))
            ring_dps = outer_dps[:, inner_idx:]
        else:
            outer_idx = int(self.thres_outer * all_size)
            inner_idx = int(self.thres_inner * all_size)
            ring_dps = sorted_dps[:, inner_idx:outer_idx]

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
