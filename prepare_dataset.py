import os
from pathlib import Path
from pdf2image import convert_from_path
import argparse

def convert_pdfs_to_images(pdf_root, output_root, dpi=200):
    os.makedirs(output_root, exist_ok=True)

    for customer_folder in os.listdir(pdf_root):
        customer_path = os.path.join(pdf_root, customer_folder)
        if not os.path.isdir(customer_path):
            continue  # skip files, only process folders
        
        output_customer_dir = os.path.join(output_root, customer_folder)
        os.makedirs(output_customer_dir, exist_ok=True)

        for file in os.listdir(customer_path):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(customer_path, file)
                print(f"Processing {pdf_path} ...")
                
                # Convert PDF to list of images
                try:
                    pages = convert_from_path(pdf_path, dpi=dpi)
                except Exception as e:
                    print(f"❌ Failed to process {pdf_path}: {e}")
                    continue

                for page_idx, page in enumerate(pages):
                    # Save each page as image
                    image_name = f"{Path(file).stem}_page{page_idx}.jpg"
                    save_path = os.path.join(output_customer_dir, image_name)
                    page.save(save_path, "JPEG")

    print(f"\n✅ PDF conversion completed. Images are saved under `{output_root}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert customer PDFs into images for training.")
    parser.add_argument("--pdf_root", type=str, default="Data", help="Root folder containing customer PDFs")
    parser.add_argument("--output_root", type=str, default="Datasets", help="Output folder to store images")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF to image conversion (default=200)")
    args = parser.parse_args()

    convert_pdfs_to_images(args.pdf_root, args.output_root, args.dpi)





### Convert PDFs to Images:


# python prepare_dataset.py --pdf_root Data --output_root Datasets --dpi 200
