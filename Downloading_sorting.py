#!/bin/env python
"""
Script for downloading SDSS galaxy image cutouts in parallel using multi-threading.
"""


import os
import logging
import concurrent.futures
import random
import time
import requests
import urllib
import pandas as pd
import csv
from astropy import units as u

def get_SDSS_url(ra, dec, impix=256, imsize=1*u.arcmin):
    """
    Generate SDSS image cutout URL based on RA, DEC, and image parameters.
    """
    base_url = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
    query_params = urllib.parse.urlencode({
        'ra': ra,
        'dec': dec,
        'width': impix,
        'height': impix,
        'scale': imsize.to(u.arcsec).value / impix
    })
    return f"{base_url}?{query_params}"

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    lst = list(lst)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_url(entry):
    """
    Download the image from the URL and save it to the specified path.
    Retries up to 5 times on failure, with increasing timeout.
    """
    downloaded = False
    tries = 5
    count = 0
    uri, path = entry
    timeout = global_timeout

    time.sleep(random.randint(1, 101) / 20.)

    while not downloaded and count < tries:
        try:
            response = requests.get(uri, stream=True, timeout=timeout)
            if response.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in response:
                        f.write(chunk)
                return path
            else:
                count += 1
                log.warning(f"Status code {response.status_code} on attempt {count}/{tries}.")
        except Exception as e:
            count += 1
            timeout *= 2
            log.warning(f"Exception on attempt {count}/{tries}: {e}. Increasing timeout to {timeout}s")
    
    return None

def multiobject_download(url_list, download_directory, log, filenames, timeout=180, concurrent_downloads=10):
    """
    Download multiple images concurrently and return the local paths of downloaded images.
    """
    global global_timeout
    #global log
    #log = log   
    global_timeout = float(timeout)

    tasks = [[url, os.path.join(download_directory, filename)] for url, filename in zip(url_list, filenames)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_url, tasks)

    downloaded_paths = [path for path in results if path]

    return downloaded_paths

# Define your paths here
if __name__ == "__main__":
    csv_path = r"C:\Users\LENOVO\Downloads\rings_gt_90.csv"
    output_dir = "output_directory"
    download_list = os.path.join(output_dir, "downloaded.csv")
    url_notfound = os.path.join(output_dir, "urlnotfound.csv")
    download_dir = os.path.join(output_dir, "panstamps")

    # Create necessary directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Initialize logger
    log = logging.getLogger("my_logger")

    # Read catalogue file into dataframe
    data = pd.read_csv(csv_path)
    racol = "ra"
    deccol = "dec"
    objidcol = "objid"

    retrieved_ids = []

    # Check if there are previously downloaded images
    try:
        downloaded = pd.read_csv(download_list, names=['paths'], dtype={"paths": str})
        paths = list(downloaded["paths"])
        retrieved_ids = [os.path.basename(x).rstrip(".jpeg") for x in paths if isinstance(x, str)]
    except Exception as e:
        print(f"Fresh download... Error: {e}")

    # Process the data in batches
    batches = list(chunks(data[[racol, deccol, objidcol]].itertuples(index=False), 10))

    for batch in batches:
        concurrent_downloads = len(batch)
        obj_ids = []
        urls = []

        for row in batch:
            RA, DEC, oid = row
            if str(oid) in retrieved_ids:
                continue

            obj_ids.append(f"{oid}.jpeg")
            try:
                url = get_SDSS_url(ra=RA, dec=DEC)
                urls.append(url)
            except Exception as e:
                with open(url_notfound, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([f"{RA},{DEC} not found"])
        
        # Download images and measure time taken
        start_time = time.time()
        local_urls = multiobject_download(
            url_list=urls,
            download_directory=download_dir,
            log=log,
            filenames=obj_ids,
            timeout=180,
            concurrent_downloads=concurrent_downloads
        )
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time}s")

        # Append downloaded paths to the list
        with open(download_list, "a") as csvfile:
            writer = csv.writer(csvfile)
            for item in local_urls:
                writer.writerow([item])

import shutil

def redistribute_images(download_dir, train_dir, test_dir, train_ratio=0.7):
    """
    Move images from the download directory to train and test directories based on the specified ratio.
    """
    images = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))]
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    
    # Shuffle images to ensure random distribution
    random.shuffle(images)
    
    train_images = images[:train_count]
    test_images = images[train_count:]
    
    for img in train_images:
        shutil.move(os.path.join(download_dir, img), os.path.join(train_dir, img))
    
    for img in test_images:
        shutil.move(os.path.join(download_dir, img), os.path.join(test_dir, img))

# Define paths to train and test directories
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Redistribute images
redistribute_images(download_dir, train_dir, test_dir, train_ratio=3398/4855)

# Function to count images in a directory
def count_images_in_folder(folder_path):
    count = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            count += 1
    return count

# Count images in train and test directories
num_train_images = count_images_in_folder(train_dir)
num_test_images = count_images_in_folder(test_dir)

print(f"Number of training images: {num_train_images}")
print(f"Number of test images: {num_test_images}")

