def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    attn_bias = attn_bias.to('cuda')
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = attn_weight.to("cuda")
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base=512, use_xpos=True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t, scale=1.):
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


def l2norm(t):
    return F.normalize(t, dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim_head=64, heads=8, dropout=0.2, forward_expansion=2, device="cuda"):
        super(TransformerBlock, self).__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.embed_dim = heads * dim_head
        self.device = device

        self.qkv = nn.Linear(dim_head * heads, dim_head * heads * 3)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.rotary_emb = RotaryEmbedding(dim_head)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim_head * heads * forward_expansion

        self.norm = nn.LayerNorm(dim_head * heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_head * heads, forward_expansion * dim_head * heads * 2),  # *2 for swiglu
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * dim_head * heads, dim_head * heads),
        )

    def forward(self, x):
        N = x.shape[0]
        seq_length = x.shape[1]
        qkv_proj = self.qkv(x)

        qkv_proj = qkv_proj.reshape(N, seq_length, self.heads, 3 * self.dim_head)
        qkv = qkv_proj.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        positions, scale = self.rotary_emb(seq_length, self.device)
        q = apply_rotary_pos_emb(positions, q, scale)
        k = apply_rotary_pos_emb(positions, k, scale ** -1)

        attention_output = scaled_dot_product_attention(q, k, v)
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.reshape(N, seq_length, self.embed_dim)

        attention_output = self.norm(attention_output)
        forward_output = self.feed_forward(attention_output)
        return attention_output + forward_output
