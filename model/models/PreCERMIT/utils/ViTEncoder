class ViTEncoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, dim=768, depth=12, heads=8, device="cuda"):
        super(ViTEncoder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, dim))

        self.transformer = nn.ModuleList([TransformerBlock(dim_head=dim // heads, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        p = self.patch_size
        bs, cnt, c, h, v = img.shape

        img = img.reshape(-1, 3, 512, 512)
        print(img.shape, "img")
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        x += self.position_embeddings
        for block in self.transformer:
            x = block(x)

        x = self.norm(x)
        print(x.shape, "do mean")
        x = x.reshape(bs, cnt, 1024, -1)
        print(x.shape, "do mean, reshape")
        x = x.mean(dim=1)
        print(x.shape, "posle mean")
        return x
