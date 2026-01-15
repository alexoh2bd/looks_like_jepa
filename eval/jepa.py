import torch
import timm 
from torchvision.transforms import v2
from torchvision.ops import MLP
from datasets import load_dataset
import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np


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
        # proj: (N*V, D) - flattened batch of projections
        A = torch.randn(proj.size(-1), 256, device=proj.device, dtype=proj.dtype) # (D, 256)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t # (N*V, 256, 1) * (knots,) -> (N*V, 256, knots)
        # err: Mean over batch (dim 0) -> (256, knots)
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * proj.size(0) # (256,) * scalar -> scalar (after mean)
        return statistic.mean()

def LeJEPA(global_proj, all_proj, sigreg, lamb, global_step=None):
    """
    global_proj: (N, Vg, D) - Embeddings of global views
    all_proj: (N, V, D) - Embeddings of all views (global + local)
    lamb: scalar weight
    """
    # Centers from global views
    centers = global_proj.mean(dim=1, keepdim=True) # (N, 1, D)
    
    # Prediction loss (MSE between centers and all views)
    # (N, 1, D) - (N, V, D) -> (N, V, D) -> scalar mean
    sim_loss = (centers - all_proj).square().mean()
    
    # SIGReg loss - applied per view, then averaged
    # all_proj is (N, V, D)
    sigreg_losses = []
    for i in range(all_proj.shape[1]):
        view_emb = all_proj[:, i, :] # (N, D)
        l = sigreg(view_emb) # scalar
        sigreg_losses.append(l)
    
    sigreg_loss = torch.stack(sigreg_losses).mean()
    
    return (1 - lamb) * sim_loss + lamb * sigreg_loss, sim_loss, sigreg_loss


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V_global=2, V_local=4, device="cuda", global_img_size=224, local_img_size=96):
        self.V_global = V_global
        self.V_local = V_local
        self.split = split
        
        self.inet_dir = "/home/users/aho13/jepa_tests/data/cache/datasets--clane9--imagenet-100/snapshots/0519dc2f402a3a18c6e57f7913db059215eee25b/data/"
        filenames = {
            "train": self.inet_dir + "train-*.parquet",
            "val": self.inet_dir + "validation*.parquet",
        }
        self.ds = load_dataset("parquet", data_files=filenames, split=split)
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        # Global Views: 224x224
        self.global_transform = v2.Compose([
            v2.RandomResizedCrop(self.global_img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
        ])
        
        # Local Views: 96x96
        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.local_img_size, scale=(0.05, 0.4)),
            v2.ToImage(),
        ])

        # Test transform
        self.test = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __getitem__(self, i):
        entry = self.ds[i]
        img = entry["image"].convert("RGB")
        label = entry["label"]
        
        if self.split == 'train':
            # Global Views
            views = []
            if self.V_global > 0:
                views += [self.global_transform(img) for _ in range(self.V_global)]
            
            # Local Views
            if self.V_local > 0:
                views += [self.local_transform(img) for _ in range(self.V_local)]
            
            return views, label
        else:
            # (C, 224, 224)
            return [self.test(img)], label

    def __len__(self):
        return len(self.ds)

class LaplacianDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split, 
        V_global=2, 
        V_local=4, 
        V_local_candidates=12,  # Generate more candidates, select best
        device="cuda", 
        global_img_size=224, 
        local_img_size=96,
        use_view_selection=True  # Enable/disable selection
    ):
        self.V_global = V_global
        self.V_local = V_local
        self.V_local_candidates = V_local_candidates
        self.split = split
        self.use_view_selection = use_view_selection and split == 'train'
        
        self.inet_dir = "/home/users/aho13/jepa_tests/data/cache/datasets--clane9--imagenet-100/snapshots/0519dc2f402a3a18c6e57f7913db059215eee25b/data/"
        filenames = {
            "train": self.inet_dir + "train-*.parquet",
            "val": self.inet_dir + "validation*.parquet",
        }
        self.ds = load_dataset("parquet", data_files=filenames, split=split)
        self.global_img_size = global_img_size
        self.local_img_size = local_img_size
        
        # Setup Laplacian kernel for view selection (on CPU, will move to GPU when needed)
        # Note: We now use OpenCV for selection, so no kernel needed here.
        
        # Global Views: 224x224
        self.global_transform = v2.Compose([
            v2.RandomResizedCrop(self.global_img_size, scale=(0.08, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # Scale to [0, 1]
        ])
        
        # Local Views: 96x96 - crop only (for selection) - Output PIL
        self.geo_crop = v2.RandomResizedCrop(
            self.local_img_size, scale=(0.05, 0.4), antialias=True
        )
        
        # Transform for selected views -> Tensor
        self.tensor_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        # Augmentation to apply after selection
        self.local_augment = v2.RandomHorizontalFlip()

        # Test transform
        self.test = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def select_best_local_views(self, img):
        """
        Generate candidate local views and select best ones based on information content
        
        Args:
            img: PIL Image
        
        Returns:
            selected_views: list of V_local best views as tensors
        """
        # Generate candidate crops (PIL Images)
        candidates = [self.geo_crop(img) for _ in range(self.V_local_candidates)]
        
        # Score using OpenCV (faster than PyTorch conv2d on CPU for this batch size)
        scores = []
        for c in candidates:
            # Convert to grayscale numpy array
            arr = np.array(c.convert("L"))
            # Variance of Laplacian
            score = cv2.Laplacian(arr, cv2.CV_64F).var()
            scores.append(score)
        
        # Select top-k indices
        # Use simple sort
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.V_local]
        
        # Apply augmentation and tensor conversion to selected views
        selected_views = [
            self.local_augment(self.tensor_transform(candidates[idx]))
            for idx in top_indices
        ]
        
        return selected_views
    
    def __getitem__(self, i):
        entry = self.ds[i]
        img = entry["image"].convert("RGB")
        label = entry["label"]
        
        if self.split == 'train':
            views = []
            
            # Global Views
            if self.V_global > 0:
                views += [self.global_transform(img) for _ in range(self.V_global)]
            
            # Local Views with selection
            if self.V_local > 0:
                if self.use_view_selection:
                    views += self.select_best_local_views(img)
                else:
                    # Original behavior: random local views
                    views += [self.tensor_transform(self.geo_crop(img)) for _ in range(self.V_local)]
                    views[-self.V_local:] = [self.local_augment(v) for v in views[-self.V_local:]]
            
            return views, label
        else:
            # (C, 224, 224)
            return [self.test(img)], label

    def __len__(self):
        return len(self.ds)
class CrossInstanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        V_global=2,
        V_local=3,
        V_mixed=1,
        global_img_size=224,
        local_img_size=96,
        part_upload=False,
    ):
        self.split = split
        self.V_global = V_global
        self.V_local = V_local
        self.V_mixed = V_mixed

        inet_dir = "/home/users/aho13/jepa_tests/data/cache/datasets--clane9--imagenet-100/snapshots/0519dc2f402a3a18c6e57f7913db059215eee25b/data/"

        filenames = (
            {"train": inet_dir + "train-00000-of-00017.parquet"}
            if part_upload
            else {
                "train": inet_dir + "train-*.parquet",
                "val": inet_dir + "validation*.parquet",
            }
        )

        self.ds = load_dataset("parquet", data_files=filenames, split=split)

        # ---- FAST label → indices mapping (vectorized) ----
        labels = self.ds["label"]  # zero-copy Arrow column
        num_classes = max(labels) + 1

        self.label_to_indices = [[] for _ in range(num_classes)]
        for idx, lbl in enumerate(labels):
            self.label_to_indices[lbl].append(idx)

        # ---- transforms ----
        self.global_transform = v2.Compose([
            v2.RandomResizedCrop(global_img_size, scale=(0.08, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
        ])

        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(local_img_size, scale=(0.05, 0.4)),
            v2.ToImage(),
        ])

        self.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.bfloat16, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __getitem__(self, i):
        entry = self.ds[i]
        img = entry["image"].convert("RGB")
        label = entry["label"]

        if self.split != "train":
            return [self.test_transform(img)], label

        # ---- preallocate ----
        total_views = self.V_global + self.V_local + self.V_mixed
        views = [None] * total_views
        k = 0

        # ---- global views ----
        for _ in range(self.V_global):
            views[k] = self.global_transform(img)
            k += 1

        # ---- local views (same image) ----
        for _ in range(self.V_local):
            views[k] = self.local_transform(img)
            k += 1

        # ---- mixed views (cross-instance) ----
        class_indices = self.label_to_indices[label]

        for _ in range(self.V_mixed):
            j = i
            while j == i:
                j = random.choice(class_indices)
            mixed_img = self.ds[j]["image"].convert("RGB")
            views[k] = self.local_transform(mixed_img)
            k += 1

        return views, label

    def __len__(self):
        return len(self.ds)


class LargeEncoder(nn.Module):
    """
    Encoder for ViT.
    """
    def __init__(self, model_name="vit_large_patch14_reg4_dinov2.lvd142m", proj_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0, 
            drop_path_rate=0.1,
            scriptable=True,  # Necessary for some versions depending on how checkpointing is implemented
            dynamic_img_size=True, # Enable variable resolution inputs (96x96 support)
        )
        self.backbone.set_grad_checkpointing(True) # Disable gradient checkpointing to fix Metadata mismatch error
        feature_dim = self.backbone.num_features 
        self.feat_dim = feature_dim  # FIXED: Store for probe initialization
        
        self.proj = MLP(
            in_channels=feature_dim, 
            hidden_channels=[2048, 2048, proj_dim], 
            norm_layer=nn.BatchNorm1d
        )
        self.output_bn = nn.BatchNorm1d(proj_dim, affine=False)

    def forward(self, x_list, unnorm=False):
        """
        x_list: List of tensors. Each tensor is (B, C, H, W).
                Can contain any mix of resolutions.
        """
        idx_global = [i for i, t in enumerate(x_list) if t.shape[-1] >= 128]
        idx_local = [i for i, t in enumerate(x_list) if t.shape[-1] < 128]
        
        results = {}

        # Process Global Views
        if len(idx_global) > 0:
            # (N, Vg, C, H, W)
            g_imgs = torch.stack([x_list[i] for i in idx_global], dim=1)
            N, Vg, C = g_imgs.shape[:3]
            
            # Forward Pass
            # backbone(flattened) -> (N*Vg, feature_dim)
            emb_g = self.backbone(g_imgs.flatten(0, 1)) 
            proj_g = self.proj(emb_g) # (N*Vg, proj_dim)
            proj_g = self.output_bn(proj_g)
            
            # Reshape back to (N, Vg, D)
            proj_g = proj_g.reshape(N, Vg, -1)
            emb_g = emb_g.reshape(N, Vg, -1)
            
            # Store results back in the dictionary
            for k, original_idx in enumerate(idx_global):
                results[original_idx] = (emb_g[:, k], proj_g[:, k])

        # Process Local Views
        if len(idx_local) > 0:
            l_imgs = torch.stack([x_list[i] for i in idx_local], dim=1)
            N, Vl, C = l_imgs.shape[:3]
            
            # backbone(flattened) -> (N*Vl, feature_dim)
            emb_l = self.backbone(l_imgs.flatten(0, 1))
            proj_l = self.proj(emb_l) # (N*Vl, proj_dim)
            proj_l = self.output_bn(proj_l)
            
            # Reshape back to (N, Vl, D)
            proj_l = proj_l.reshape(N, Vl, -1)
            emb_l = emb_l.reshape(N, Vl, -1)
            
            for k, original_idx in enumerate(idx_local):
                results[original_idx] = (emb_l[:, k], proj_l[:, k])

        # Reassemble in original order
        final_emb = []
        final_proj = []
        
        for i in range(len(x_list)):
            e, p = results[i]
            final_emb.append(e)
            final_proj.append(p)
        
        # Stack to return (N, V, D)
        return torch.stack(final_emb, dim=1), torch.stack(final_proj, dim=1)
