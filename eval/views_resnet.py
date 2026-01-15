import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm
import wandb
import hydra
import tqdm
import logging
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

# Setup logging
logging.basicConfig(level=logging.INFO)

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: (V, N, D)
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class Encoder(nn.Module):
    def __init__(self, model_name="resnet50.a1_in1k", proj_dim=512):
        super().__init__()
        # FIX: Use num_classes=0 to get the 2048-dim feature pool, not a random 512-dim projection
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0, 
            drop_path_rate=0.1,
        )
        # ResNet50 output is 2048, ResNet18 is 512. Assuming ResNet50:
        feature_dim = self.backbone.num_features 
        
        self.proj = MLP(
            in_channels=feature_dim, 
            hidden_channels=[2048, 2048, proj_dim], 
            norm_layer=nn.BatchNorm1d
        )
        self.output_bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, x_list):
        """
        x_list: List of tensors. Each tensor is (B, C, H, W).
                Can contain any mix of resolutions.
        """
        # 1. Sort indices by resolution
        #    We assume 224 (or larger) is "Global" and anything else is "Local"
        #    Adjust 128 as the cutoff threshold if needed.
        idx_global = [i for i, t in enumerate(x_list) if t.shape[-1] >= 128]
        idx_local = [i for i, t in enumerate(x_list) if t.shape[-1] < 128]
        
        # Prepare a dictionary to store results by their original index
        # key: index in x_list, value: (embedding, projection)
        results = {}

        # 2. Process Global Views (if any exist)
        if len(idx_global) > 0:
            # Stack them: (B, V_g, C, H, W)
            g_imgs = torch.stack([x_list[i] for i in idx_global], dim=1)
            N, Vg, C, H, W = g_imgs.shape
            
            # Forward Pass
            # flatten(0,1) merges Batch and View dims for the backbone
            emb_g = self.backbone(g_imgs.flatten(0, 1)) # (N*Vg, 2048)
            proj_g = self.proj(emb_g)
            proj_g = self.output_bn(proj_g)
            # proj_g = F.normalize(proj_g, dim=-1)  # Add this
            proj_g = proj_g.reshape(N, Vg, -1)
            
            # Store results back in the dictionary
            emb_g = emb_g.reshape(N, Vg, -1) # Unflatten for storage
            for k, original_idx in enumerate(idx_global):
                results[original_idx] = (emb_g[:, k], proj_g[:, k])

        # 3. Process Local Views (if any exist)
        if len(idx_local) > 0:
            l_imgs = torch.stack([x_list[i] for i in idx_local], dim=1)
            N, Vl, C, H, W = l_imgs.shape
            
            emb_l = self.backbone(l_imgs.flatten(0, 1))
            proj_l = self.proj(emb_l)
            proj_l = self.output_bn(proj_l)
            # proj_l = F.normalize(proj_l, dim=-1)  # Add this

            proj_l = proj_l.reshape(N, Vl, -1)
            
            emb_l = emb_l.reshape(N, Vl, -1)
            for k, original_idx in enumerate(idx_local):
                results[original_idx] = (emb_l[:, k], proj_l[:, k])

        # 4. Reassemble the list in the original order
        #    This ensures net(vs)[0] matches vs[0]
        final_emb = []
        final_proj = []
        
        for i in range(len(x_list)):
            e, p = results[i]
            final_emb.append(e)
            final_proj.append(p)
            
        # Stack to return (N, V, D)
        # We assume all views have the same batch size N
        return torch.stack(final_emb, dim=1), torch.stack(final_proj, dim=1)
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        self.split = split
        
        # FIX: Explicit safe directory or config usage
        self.inet_dir = "/home/users/aho13/jepa_tests/data/cache/datasets--clane9--imagenet-100/snapshots/0519dc2f402a3a18c6e57f7913db059215eee25b/data/"
        filenames = {
            "train": self.inet_dir + "train-*.parquet",
            "val": self.inet_dir + "validation*.parquet",
            }
        self.ds = load_dataset("parquet", data_files=filenames, split=split)
        # self.ds = Imagenette(root="../lejepa/data/imagenette", size="224px", split=split, download=False)
        
        
        # Global Views: 224x224
        self.global_transfo = v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
        ])
        
        # FIX: Cleaned up Local Transfo syntax error
        self.local_transfo = v2.Compose([
            v2.RandomResizedCrop(96, scale=(0.05, 0.4)),
            v2.ToImage(),
        ])

        # FIX: Cleaned up Test Transfo (was mixed with garbage text)
        self.test = v2.Compose([
            v2.Resize(256), # Standard Test: Resize 256 -> CenterCrop 224
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, i):
        entry = self.ds[i]
        img = entry["image"].convert("RGB")
        label = entry["label"]
        
        if self.V > 1 and self.split == 'train':
            # Global Views
            global_views = [self.global_transfo(img) for _ in range(self.n_global)]
            # Local Views
            local_views = [self.local_transfo(img) for _ in range(self.n_local)]
            return global_views + local_views, label
        else:
            return [self.test(img)], label

    def __len__(self):
        return len(self.ds)

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="JEPA_Views", config=dict(cfg))
    torch.manual_seed(0)
    num_workers = cfg.num_workers if "num_workers" in cfg else 1
    proj_dim = cfg.proj_dim if "proj_dim" in cfg else 2048
    lr = cfg.lr if "lr" in cfg else 2e-3
    bs = cfg.bs if "bs" in cfg else 256
    epochs = cfg.epochs if "epochs" in cfg else 300
    lamb = cfg.lamb if "lamb" in cfg else 0.05
    V = cfg.V if "V" in cfg else 4
    device = cfg.device if "device" in cfg else "cuda"
    num_labels= 100


    train_ds = HFDataset("train", V=V)
    test_ds = HFDataset("val", V=1) # Note: 'validation' usually, check dataset
    

    # Configure View Counts
    if V > 2: 
         train_ds.n_global = 2
         train_ds.n_local = V - 2
    else:
         train_ds.n_global = V
         train_ds.n_local = 0
         
    # Data Loaders
    # train = DataLoader(
    #     train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=True,prefetch_factor=2,
    # )
    train = DataLoader(
        train_ds, 
        batch_size=bs, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers,          # <--- Critical
        pin_memory=True, 
        persistent_workers=True, # <--- Keeps workers alive between epochs
        prefetch_factor=4        # <--- Prepare 4 batches ahead per worker
    )
    test = DataLoader(test_ds, batch_size=bs, num_workers=num_workers, pin_memory=True, persistent_workers=True,prefetch_factor=2,)

    # Model Setup
    net = Encoder(proj_dim=proj_dim).to(device)
    
    probe = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, num_labels)).to(device)
    sigreg = SIGReg().to(device)

    # Optimizer
    g1 = {"params": net.parameters(), "lr": lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 3e-3}
    opt = torch.optim.SGD([g1], momentum=0.9, nesterov=True)
    opt_probe = torch.optim.Adam([g2], lr=3e-3)

    warmup_steps = len(train) # 1 epoch warmup
    total_steps = len(train) * epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    global_step = 0
    

    gpu_aug_global = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        # Large Kernel OK for 224px
        v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
        v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2), 
        v2.ToDtype(torch.bfloat16, scale=True), # Scale 0-255 -> 0-1 BEFORE Normalize
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

    gpu_aug_local = v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        # FIX: Smaller Kernel for 96px images (approx 10% of size)
        v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        v2.ToDtype(torch.bfloat16, scale=True), # Scale 0-255 -> 0-1 BEFORE Normalize
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)
    
    # Training
    for epoch in range(epochs):
        net.train(), probe.train()
        logging.info(f"Epoch {epoch}")
        
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with autocast(device, dtype=torch.bfloat16):
                # vs is a list of tensors from DataLoader if collate_default does its job on list-returning-dataset
                # vs: [Tensor(B, C, H_g, W_g), Tensor(B, C, H_g, W_g), Tensor(B, C, H_l, W_l)...]
                
                # Move each to cuda
                vs = [v.to(device, non_blocking=True) for v in vs]

                # Augment in cuda
                y = y.to(device, non_blocking=True)
                g_imgs = [vs[i] for i in range(V) if vs[i].shape[-1] == 224]
                l_imgs = [vs[i] for i in range(V) if vs[i].shape[-1] == 96]
                g_imgs = [gpu_aug_global(g) for g in g_imgs]
                l_imgs = [gpu_aug_local(l) for l in l_imgs]
                views = g_imgs + l_imgs
                # Forward
                emb, proj = net(views) # proj: (V, N, D)
                inv_loss = (proj.mean(1,keepdim=True) - proj).square().mean()
                sigreg_loss = sigreg(proj.flatten(0,1))
                lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
                
                # Probe Loss
                # Flatten emb for probe: (N*V, 2048)
                y_rep = y.repeat_interleave(len(vs))
                yhat = probe(emb.flatten(0, 1).detach()) 
                
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            opt_probe.zero_grad()
            loss.backward()
            opt.step()
            opt_probe.step()
            scheduler.step()
            
            if global_step %20 ==0:
                wandb.log(
                    {
                        "train/probe_loss": probe_loss.item(),
                        "train/lejepa_loss": lejepa_loss.item(),
                        "train/sigreg_loss": sigreg_loss.item(),
                        "train/invariance_loss": inv_loss.item(),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/projected_norm": proj.norm().item(),
                    },
                    step=global_step,  
                )
            global_step += 1

        # Evaluation
        net.eval(), probe.eval()
        correct = 0
        total_samples = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = [v.to("cuda", non_blocking=True) for v in vs]
                y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    # net(vs)[0] is emb: (N, 1, 2048)
                    # Use view 0
                    logits = probe(net(vs)[0][:, 0])
                    correct += (logits.argmax(1) == y).sum().item()
                    total_samples += y.size(0)
                    
        acc = correct / total_samples
        wandb.log({"test/acc": acc, "test/epoch": epoch})
        logging.info(f"Epoch {epoch} Test Acc: {acc:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()