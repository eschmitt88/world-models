"""Training-throughput probe for small causal (block-causal) DiTs on the local GPU.
Usage: python scripts/bench_dit.py "(tok_per_frame, frames, batch, patch, steps, use_flex)"
Run each config in a fresh process (an OOM leaves memory held). Results: ADR 0001."""
import torch, time, math
import torch.nn as nn, torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex=torch.compile(flex_attention)
torch.backends.cuda.matmul.allow_tf32=True
class Block(nn.Module):
    def __init__(s,d,h):
        super().__init__(); s.h=h; s.n1=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d); s.n2=nn.LayerNorm(d)
        s.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); s.ada=nn.Linear(d,6*d)
    def forward(s,x,c,mask):
        a,b,g,a2,b2,g2=s.ada(c).chunk(6,-1)
        y=s.n1(x)*(1+a)+b; B,L,D=y.shape
        q,k,v=s.qkv(y).view(B,L,3,s.h,D//s.h).permute(2,0,3,1,4)
        y=(flex(q,k,v,block_mask=mask) if not isinstance(mask,torch.Tensor) else F.scaled_dot_product_attention(q,k,v,attn_mask=mask)).transpose(1,2).reshape(B,L,D)
        x=x+g*s.o(y); y=s.n2(x)*(1+a2)+b2; return x+g2*s.ff(y)
class DiT(nn.Module):
    def __init__(s,d=512,depth=10,h=8,pdim=192):
        super().__init__(); s.inp=nn.Linear(pdim,d); s.blocks=nn.ModuleList([Block(d,h) for _ in range(depth)]); s.out=nn.Linear(d,pdim); s.t=nn.Linear(1,d)
    def forward(s,x,t,mask):
        h=s.inp(x); c=s.t(t)
        for b in s.blocks: h=b(h,c,mask)
        return s.out(h)
def run(tok_per_frame, frames, B, patch, steps=20, use_flex=True):
    pdim=3*patch*patch; L=tok_per_frame*frames
    m=DiT(pdim=pdim).cuda(); opt=torch.optim.AdamW(m.parameters(),1e-4)
    fid=torch.arange(frames).repeat_interleave(tok_per_frame).cuda()
    if use_flex:
        mask=create_block_mask(lambda b,h,q,kv: fid[kv]<=fid[q], None, None, L, L)
    else:
        mask=(fid[None,:]<=fid[:,None])  # block-causal
    nparam=sum(p.numel() for p in m.parameters())/1e6
    x=torch.randn(B,L,pdim,device='cuda'); t=torch.rand(B,L,1,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps+5):
        if i==5: torch.cuda.synchronize(); t0=time.time()
        with torch.autocast('cuda',torch.bfloat16):
            loss=F.mse_loss(m(x,t,mask),x)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt=(time.time()-t0)/steps
    print(f"params {nparam:.1f}M patch{patch} {tok_per_frame}tok/frame x {frames} frames = {L} tok, batch {B}: {B/dt:.1f} seq/s, {B*L/dt/1e3:.0f}k tok/s, {B*frames/dt:.0f} frames/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    del m,opt
    torch.cuda.empty_cache()
import sys, ast
run(*ast.literal_eval(sys.argv[1]))
