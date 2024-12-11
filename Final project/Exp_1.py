import math
import time
import os
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pillow_heif import register_heif_opener
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from metrics import *


register_heif_opener()

# Directory containing input images
input_dir = "./image_set"

# Collect all image files from the input directory
input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.tiff'))]

if not input_files:
    raise FileNotFoundError(f"No valid image files found in {input_dir}.")


def compress(quality_jp=None, quality_jp2=None, quality_heif=None):
    results = {}
    methods = []

    # Process each image
    for input_path in input_files:
        image_name = os.path.basename(input_path)
        results[image_name] = {}
        # Load the original image
        original_img = Image.open(input_path).convert("RGB")
        original_arr = np.array(original_img)
        original_size = os.path.getsize(input_path)  # in bytes
        img1_ts = torch.from_numpy(original_arr)
        img1_ts = img1_ts.unsqueeze(0)

        if quality_jp is not None:
            # --- JPEG Compression ---
            methods.append("JPEG")
            jpeg_path = f"output_jpeg/{os.path.basename(input_path)}_q{quality_jp}.jpg"
            start_time = time.time()
            original_img.save(jpeg_path, format="JPEG", quality=quality_jp)
            jpeg_time = time.time() - start_time
            jpeg_size = os.path.getsize(jpeg_path)
            jpeg_ratio = jpeg_size / original_size
            jpeg_recon = Image.open(jpeg_path).convert("RGB")
            jpeg_recon_arr = np.array(jpeg_recon)

            jpeg_ssim = ssim(original_arr, jpeg_recon_arr, multichannel=True)
            jpeg_mse = mse_metric(original_arr, jpeg_recon_arr)
            jpeg_psnr = psnr_metric(jpeg_mse, max_val=255.0)

            results[image_name]["JPEG"] = {
                "Time": jpeg_time,
                "Ratio": jpeg_ratio,
                "SSIM": jpeg_ssim,
                "MSE": jpeg_mse,
                "PSNR": jpeg_psnr
            }

        if quality_jp2 is not None:
            # --- JPEG2000 Compression ---
            methods.append("JPEG2000")
            jp2_path = f"output_jpeg2000/{os.path.basename(input_path)}_q{quality_jp2}.jp2"
            start_time = time.time()
            original_img.save(jp2_path, format="JPEG2000", quality_mode="rates", quality_layers=[quality_jp2])
            jp2_time = time.time() - start_time
            jp2_size = os.path.getsize(jp2_path)
            jp2_ratio = jp2_size / original_size
            jp2_recon = Image.open(jp2_path).convert("RGB")
            jp2_recon_arr = np.array(jp2_recon)

            # evaluate quality
            jp2_ssim = ssim(original_arr, jp2_recon_arr, multichannel=True)
            jp2_mse = mse_metric(original_arr, jp2_recon_arr)
            jp2_psnr = psnr_metric(jp2_mse, max_val=255.0)

            results[image_name]["JPEG2000"] = {
                "Time": jp2_time,
                "Ratio": jp2_ratio,
                "SSIM": jp2_ssim,
                "MSE": jp2_mse,
                "PSNR": jp2_psnr
            }

        if quality_heif is not None:
            # --- HEIF Compression ---
            methods.append("HEIF")
            heif_path = f"output_heif/{os.path.basename(input_path)}_q{quality_heif}.heif"
            start_time = time.time()
            original_img.save(heif_path, format="HEIF", quality=quality_heif)
            heif_time = time.time() - start_time
            heif_size = os.path.getsize(heif_path)
            heif_ratio = heif_size / original_size
            heif_recon = Image.open(heif_path).convert("RGB")
            heif_recon_arr = np.array(heif_recon)

            heif_ssim = ssim(original_arr, heif_recon_arr, multichannel=True)
            heif_mse = mse_metric(original_arr, heif_recon_arr)
            heif_psnr = psnr_metric(heif_mse, max_val=255.0)

            results[image_name]["HEIF"] = {
                "Time": heif_time,
                "Ratio": heif_ratio,
                "SSIM": heif_ssim,
                "MSE": heif_mse,
                "PSNR": heif_psnr
            }

    print("=== RESULTS ===")
    for img_name, data in results.items():
        print(f"Image: {img_name}")
        for method in methods:
            stats = data[method]
            print(f"  {method}:")
            print(f"    Time (s): {stats['Time']:.4f}")
            print(f"    Size Ratio (compressed/original): {stats['Ratio']:.4f}")
            print(f"    SSIM: {stats['SSIM']:.4f}")
            print(f"    MSE: {stats['MSE']:.4f}")
            print(f"    PSNR (dB): {stats['PSNR']:.4f}")
        print()


if __name__ == "__main__":
    Quality_jp = None
    Quality_jp2 = None
    Quality_heif = None
    compress(Quality_jp, Quality_jp2, Quality_heif)

