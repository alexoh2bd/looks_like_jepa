
from torchvision.transforms import v2

def gpu_aug_global(device="cuda"):
    return v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        # Large Kernel OK for 224px
        v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
        v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2), 
        v2.ToDtype(torch.bfloat16, scale=True), # Scale 0-255 -> 0-1 BEFORE Normalize
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

def gpu_aug_local(device="cuda"):
    return  v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        # FIX: Smaller Kernel for 96px images (approx 10% of size)
        v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        v2.ToDtype(torch.bfloat16, scale=True), # Scale 0-255 -> 0-1 BEFORE Normalize
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)