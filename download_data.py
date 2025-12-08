#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:02:56 2025

@author: fmry
"""
    
import os
import urllib.request
import zipfile

import gdown

def download_celeba_hq(dest="/work3/fmry/Data/coco/"):
    os.makedirs(dest, exist_ok=True)
    
    file_id = "188K19ucknC6wg1R6jbuPEhTq9zoufOx4"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    zip_path = os.path.join(dest, "celeba-hq.zip")
    
    print("Downloading CelebA-HQ...")
    # Important: set use_cookies=False
    gdown.download(url, zip_path, quiet=False, use_cookies=False)
    
    # Extract if zip
    if zipfile.is_zipfile(zip_path):
        print("Extracting CelebA-HQ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest)
        print("Extraction complete!")
    else:
        print("Download complete! (not a zip file)")
    
    print("CelebA-HQ is ready at:", dest)


def download_and_extract(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, os.path.basename(url))

    if not os.path.exists(zip_path):
        print("Downloading:", url)
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete:", zip_path)

    print("Extracting:", zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_folder)
    print("Extraction complete.")


def download_coco2017(dest="/work3/fmry/Data/coco/"):
    os.makedirs(dest, exist_ok=True)

    # official COCO URLs
    urls = [
        # images
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",

        # annotations
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ]

    for url in urls:
        download_and_extract(url, dest)

    print("COCO 2017 ready at:", dest)

if __name__ == "__main__":
    download_celeba_hq()
    download_coco2017()
