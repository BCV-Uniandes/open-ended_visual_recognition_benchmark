import torch
import torch.nn as nn
import torch.nn.functional as F

from . import distributed as dist


class DifferentiableDistGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, max_batch_size=None):
        """
        Forward function will be the distributed.all_gather
        :param tensor: batch_size, hid_dim
        """
        ctx.input = tensor
        size = torch.tensor([tensor.shape[0]]).to(tensor)

        if max_batch_size is None:
            max_batch_size = tensor.shape[0]
        else:
            if tensor.dim() == 1:
                tmp = torch.zeros([max_batch_size]).to(tensor)
                tmp[: tensor.shape[0]] = tensor
                tensor = tmp
            else:
                tensor = F.pad(
                    tensor,
                    (0, 0, 0, max_batch_size - tensor.shape[0]),
                    value=0,
                    mode="constant",
                )

        # [GPU1's (b, hid), GPU2's (b, hid), ...]
        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        gathered_size = [
            torch.zeros([1]).to(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensor, tensor)
        torch.distributed.all_gather(gathered_size, size)
        ctx.per_gpu_batch_size = [int(x.item()) for x in gathered_size]

        # [gpu_1_0, gpu_1_1, ..., gpu_2_0, gpu_2_1, ...]
        gathered_tensor = torch.cat(gathered_tensor, 0)
        gathered_size = torch.cat(gathered_size, 0)
        # remove the padding
        mask = torch.zeros([gathered_tensor.shape[0]]).to(gathered_tensor)
        for i in range(torch.distributed.get_world_size()):
            mask[
                i * max_batch_size : (i * max_batch_size) + int(gathered_size[i].item())
            ] = 1
        gathered_tensor = gathered_tensor[mask == 1]
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        backward function will be the distributed.reduce_scatter (with sum as reduce OP)
        :param grad_output: world_size x per_gpu_batch_size, hid_dim
        """
        grad_input = torch.zeros_like(ctx.input)

        # Split the current grad through the batch dim, results in
        #   [(per_gpu_batch_size, hid_dim), ...]
        grad_output = grad_output.contiguous()
        grad_output_list = list(
            grad_output.split(split_size=ctx.per_gpu_batch_size, dim=0)
        )

        # Reduce for each tensor and scatter them.
        torch.distributed.reduce_scatter(
            grad_input,
            grad_output_list,
            op=torch.distributed.ReduceOp.SUM,
            async_op=False,
        )

        return grad_input, None


LARGE_NUM = 1e9


class DistributedContrastiveLossMultiPositive(torch.nn.Module):
    """
    Modified from the original TF implementation here:
    https://github.com/google-research/simclr/blob/master/objective.py
    """

    def forward(
        self,
        hidden1,
        hidden2,
        prompts_per_image=None,
        apply_hidden_norm=True,
        temperature=0.1,
        is_dist=True,
        is_symmetric_loss=True,
        stabilize_loss=False,
        get_accuracy=False,
        key1=None,
        key2=None,
        random_negatives=False,
        max_batch_size=100,
    ):
        """
        Calculating the contrastive loss for both sides.
        :param hidden1: Tensor[batch_size, dim]
        :param hidden2: Tensor[batch_size, dim]
        :param apply_hidden_norm: Whether need to apply norm on the hiddens.
        :param logit_scale: 1 / temperature in the softmax
        :param is_dist: Whether it is a distributed version.
        :param is_nce: Whether it is an NCE loss as in SimCLR (o.w., it is a simple contrastive loss)
        :param key1: list(batch_size)
        :param key2: list(batch_size)
        :return: loss of Tensor[0]
        """
        # Gather keys
        if key1 is None:
            key1 = key2
        if key2 is None:
            key2 = key1

        key1_hash = torch.tensor(
            [hash(key) for key in key1], device=hidden1.device, dtype=hidden1.dtype
        )
        key2_hash = torch.tensor(
            [hash(key) for key in key2], device=hidden2.device, dtype=hidden2.dtype
        )

        if is_dist:
            with torch.no_grad():
                key1_hash_large = dist.all_gather_tensor(key1_hash)
                key2_hash_large = dist.all_gather_tensor(key2_hash)

        else:
            key1_hash_large = key1_hash
            key2_hash_large = key2_hash

        # Get (normalized) hidden1 and hidden2.
        if apply_hidden_norm:
            hidden1 = F.normalize(hidden1, dim=-1)
            hidden2 = F.normalize(hidden2, dim=-1)
        batch_size = hidden1.shape[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if is_dist:
            # Gather all hiddens from other servers
            hidden1_large = dist.all_gather_tensor(hidden1)
            hidden2_large = dist.all_gather_tensor(hidden2)

            if prompts_per_image is not None:
                prompts_per_image_large = dist.all_gather_tensor(
                    prompts_per_image.to(dtype=torch.uint8)
                )
            else:
                prompts_per_image_large = None

        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            prompts_per_image_large = prompts_per_image

        if prompts_per_image is not None:
            hidden1 = hidden1[prompts_per_image]
            hidden1_large = hidden1_large[prompts_per_image_large]

        logits_ab = torch.matmul(hidden1, hidden2_large.t()) * (1 / temperature)
        logits_ba = torch.matmul(hidden2, hidden1_large.t()) * (1 / temperature)
        loss_func = torch.nn.CrossEntropyLoss()

        # If two keys are the same, we consider them as positive pairs.\
        # key2_hash_large = key2_hash_large[mask2 == 1]
        label_ab = (key1_hash.unsqueeze(1) == key2_hash_large.unsqueeze(0)).float()
        if random_negatives:
            logits_ab = logits_ab[label_ab.sum(1) > 0]
            logits_aa = torch.matmul(
                hidden1[label_ab.sum(1) > 0], hidden1[label_ab.sum(1) == 0].t()
            ) * (1 / temperature)
            label_aa = torch.zeros(logits_aa.shape).to(label_ab)
            label_ab = label_ab[label_ab.sum(1) > 0]
            label_ab = torch.cat([label_ab, label_aa], 1)
            logits_ab = torch.cat([logits_ab, logits_aa], 1)

        # We will normalize the loss by the number of positives.
        label_ab /= label_ab.sum(1, keepdim=True)

        # Do the same for the reverse pair.
        # key1_hash_large = key1_hash_large[mask1 == 1]
        label_ba = (key2_hash.unsqueeze(1) == key1_hash_large.unsqueeze(0)).float()

        # Ignore entries where all labels are 0.
        logits_ba = logits_ba[label_ba.sum(1) > 0]
        label_ba = label_ba[label_ba.sum(1) > 0]
        # We will normalize the loss by the number of positives.
        label_ba /= label_ba.sum(1, keepdim=True)
        loss_a = loss_func(logits_ab, label_ab)
        loss_b = loss_func(logits_ba, label_ba)

        if is_symmetric_loss:
            loss = (loss_a + loss_b) / 2.0
        else:
            loss = loss_a

        if stabilize_loss:
            loss = loss * temperature

        if get_accuracy:
            # Assuming logits_ab and logits_ba have the same shape as labels_prob
            # Find the indices of the predicted classes
            _, pred_ab = torch.max(logits_ab, 1)
            _, pred_ba = torch.max(logits_ba, 1)

            # Find the correct predictions by checking if any of the predicted classes match the true classes
            correct_ab = (label_ab > 0.0).gather(1, pred_ab.unsqueeze(1)).sum().item()
            correct_ba = (label_ba > 0.0).gather(1, pred_ba.unsqueeze(1)).sum().item()

            # Calculate the accuracy
            total = label_ab.size(0)
            accuracy_ab = correct_ab / total
            accuracy_ba = correct_ba / total

            return (loss.squeeze(), accuracy_ab, accuracy_ba)
        else:
            return loss


class DistributedContrastiveLoss(torch.nn.Module):
    """
    Modified from the original TF implementation here:
    https://github.com/google-research/simclr/blob/master/objective.py
    """

    def compute_logits(self, hidden1, hidden2, regions, logit_scale):
        b, n, d = hidden1.shape
        weights = hidden1 @ hidden2.T
        if regions is not None:
            weights[~regions] = -float(torch.inf)
        weights = torch.nn.functional.softmax(weights, dim=1)
        hidden1 = hidden1.reshape(-1, hidden1.shape[-1])
        weights = weights.reshape(-1, weights.shape[-1])
        hidden1 = torch.bmm(hidden1.unsqueeze(dim=-1), weights.unsqueeze(dim=1))
        hidden1 = hidden1.reshape(b, n, d, -1).sum(dim=1)  # vb, d, tb
        hidden1 = F.normalize(hidden1, dim=1)
        logits = (
            torch.bmm(hidden1.permute(2, 0, 1), hidden2.unsqueeze(dim=-1))
            .squeeze(dim=-1)
            .permute(1, 0)
        )
        return logits * logit_scale

    def forward(
        self,
        hidden1,
        hidden2,
        prompts_per_image=None,
        apply_hidden_norm=True,
        logit_scale=1.0,
        is_dist=True,
        is_nce=False,
        is_symmetric_loss=True,
        stabilize_loss=False,
        get_accuracy=False,
        max_batch_size=100,
        **kwargs
    ):
        """
        Calculating the contrastive loss for both sides.
        :param hidden1: Tensor[batch_size, dim]
        :param hidden2: Tensor[batch_size, dim]
        :param apply_hidden_norm: Whether need to apply norm on the hiddens.
        :param logit_scale: 1 / temperature in the softmax
        :param is_dist: Whether it is a distributed version.
        :param is_nce: Whether it is an NCE loss as in SimCLR (o.w., it is a simple contrastive loss)
        :return: loss of Tensor[0]
        """
        # Get (normalized) hidden1 and hidden2.
        if apply_hidden_norm:
            hidden1 = F.normalize(hidden1, dim=-1)
            hidden2 = F.normalize(hidden2, dim=-1)
        batch_size = hidden2.shape[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if is_dist:
            # Gather all hiddens from other servers
            hidden1_large = dist.all_gather([hidden1])
            hidden1_large = torch.vstack(hidden1_large)
            hidden2_large = dist.all_gather([hidden2])
            hidden2_large = torch.vstack(hidden2_large)

            if prompts_per_image is not None:
                prompts_per_image_large = dist.all_gather([prompts_per_image])
                prompts_per_image_large = torch.vstack(prompts_per_image_large)
            else:
                prompts_per_image_large = None

            enlarged_batch_size = hidden1_large.shape[0]

            # Get the part of diagonal labels for this rank.
            replica_id = torch.distributed.get_rank()
            labels_idx = (
                torch.arange(batch_size, device=hidden1.device)
                + replica_id * batch_size
            )
            masks = F.one_hot(labels_idx, enlarged_batch_size).float()
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            prompts_per_image_large = prompts_per_image
            labels_idx = torch.arange(batch_size, device=hidden1.device)
            masks = F.one_hot(
                torch.arange(batch_size, device=hidden1.device), batch_size
            ).float()

        # Calculate the logits.
        logits_ab = self.compute_logits(
            hidden1, hidden2_large, prompts_per_image, logit_scale
        )
        logits_ba = self.compute_logits(
            hidden1_large, hidden2, prompts_per_image_large, logit_scale
        ).T
        loss_func = torch.nn.CrossEntropyLoss()
        if is_nce:
            # For NCE loss, we also consider other data samples on the same side as Negative.
            logits_aa = torch.matmul(hidden1, hidden1_large.t()) * logit_scale
            logits_aa = (
                logits_aa - masks * LARGE_NUM
            )  # Exclude the Xi . Xi; Since it is positive pair.
            logits_bb = torch.matmul(hidden2, hidden2_large.t()) * logit_scale
            logits_bb = logits_bb - masks * LARGE_NUM
            loss_a = loss_func(torch.cat([logits_ab, logits_aa], 1), labels_idx)
            loss_b = loss_func(torch.cat([logits_ba, logits_bb], 1), labels_idx)
        else:
            # For contrastive loss, we only consider unaligned samples from other side as Negative.
            loss_a = loss_func(logits_ab, labels_idx)
            loss_b = loss_func(logits_ba, labels_idx)

        if is_symmetric_loss:
            loss = (loss_a + loss_b) / 2.0
        else:
            loss = loss_a

        if stabilize_loss:
            loss = loss / logit_scale

        if get_accuracy:
            _, pred_ab = torch.max(logits_ab, 1)
            _, pred_ba = torch.max(logits_ba, 1)
            correct_ab = (pred_ab == labels_idx).sum().item()
            correct_ba = (pred_ba == labels_idx).sum().item()
            total = labels_idx.size(0)
            accuracy_ab = correct_ab / total
            accuracy_ba = correct_ba / total
            return loss, accuracy_ab, accuracy_ba
        else:
            return loss
