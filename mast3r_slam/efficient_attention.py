"""Memory-bounded attention for the copied MASt3R network on 8 GB GPUs."""
import types
import torch
import torch.nn.functional as F


def _attention(module, q, k, v):
    dtype = q.dtype
    # cuRoPE and all model parameters remain FP32. Only attention uses BF16
    # (FP16 on older devices) to permit PyTorch's fused CUDA implementation.
    if q.is_cuda:
        compact = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q, k, v = (x.to(compact) for x in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0., scale=module.scale).to(dtype)


def _self_forward(self, x, xpos):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, i] for i in range(3)]
    if self.rope is not None:
        q, k = self.rope(q, xpos), self.rope(k, xpos)
    x = _attention(self, q, k, v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


def _cross_forward(self, query, key, value, qpos, kpos):
    B, N, C = query.shape
    q = self.projq(query).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
    k = self.projk(key).reshape(B, key.shape[1], self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
    v = self.projv(value).reshape(B, value.shape[1], self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
    if self.rope is not None:
        q, k = self.rope(q, qpos), self.rope(k, kpos)
    x = _attention(self, q, k, v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


def enable_efficient_attention(model):
    count = 0
    for module in model.modules():
        name = type(module).__name__
        if name in ("Attention", "CrossAttention"):
            module.forward = types.MethodType(_self_forward if name == "Attention" else _cross_forward, module)
            count += 1
    return count
