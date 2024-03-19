from lpips import LPIPS
import torch
import torch.nn.functional as F
from math import log10
from torchmetrics.functional.image import structural_similarity_index_measure
def check_range(img):
    assert torch.min(img) >= -1.0, "Minimum pixel value is less than -1.0"
    assert torch.max(img) <= 1.0, "Maximum pixel value is greater than 1.0"
    
def calculate_lpips(img1, img2):
    check_range(img1)
    check_range(img2)
    
    lpips_model = LPIPS(net="alex").to(img2.device)
    lpips = lpips_model(img1.to(img2.device), img2).item()
    return lpips

def calculate_ssim(img1, img2):
    check_range(img1)
    check_range(img2)
    
    img1 = img1.to(img2.device)
    img1_normalized = (img1 + 1.) / 2.
    img2_normalized = (img2 + 1.) / 2.
    ssim_value = structural_similarity_index_measure(img1_normalized, img2_normalized)
    return ssim_value

def calculate_psnr(img1, img2):
    check_range(img1)
    check_range(img2)
    
    img1_normalized = (img1 + 1.) / 2.
    img2_normalized = (img2 + 1.) / 2.
    mse = F.mse_loss(img1_normalized, img2_normalized)
    psnr = 20 * log10(1.0 / torch.sqrt(mse))
    return psnr

def calculate_l1(img1, img2):
    check_range(img1)
    check_range(img2)
    
    img1_normalized = (img1 + 1.) / 2.
    img2_normalized = (img2 + 1.) / 2.
 
    l1_loss = F.l1_loss(img1_normalized, img2_normalized)
    return l1_loss.item()