import asyncio
import aiohttp
import base64
import time
import tracemalloc
import json
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor
from logger import logging
from exception.exception_handing import CustomExceptionHandling
import sys
import os


def get_image_sources_from_url(url: str, delay: int = 3) -> list[str]:
    """
    Launches headless Chrome to collect all <img src="..."> links from the web page.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(options=options)
    logging.logging.info(f"🌐 Opening: {url}")
    driver.get(url)
    time.sleep(delay)  # let JavaScript finish
    img_elements = driver.find_elements(By.TAG_NAME, "img")
    img_sources = [img.get_attribute("src") for img in img_elements if img.get_attribute("src")]
    driver.quit()
    logging.logging.info(f"🖼️ Found {len(img_sources)} image sources.")
    return img_sources


def decode_base64_image(src: str) -> Image.Image | None:
    try:
        header, encoded = src.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logging.logging.info(f"❌ Failed to decode base64 image: {e}")
        return None


def decode_http_image(data: bytes) -> Image.Image | None:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        logging.logging.info(f"❌ Failed to decode image bytes: {e}")
        return None


async def fetch_image(session: aiohttp.ClientSession, src: str, idx: int, executor: ThreadPoolExecutor) -> Image.Image | None:
    if src.startswith("data:image"):
        # decode base64 image in executor (thread-safe)
        return await asyncio.get_event_loop().run_in_executor(executor, decode_base64_image, src)
    elif src.startswith("http"):
        try:
            async with session.get(src, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.read()
                    return await asyncio.get_event_loop().run_in_executor(executor, decode_http_image, data)
        except Exception as e:
            logging.logging.info(f"❌ Error fetching image {idx}: {e}")
    return None


async def extract_images_from_website_async(url: str) -> list[Image.Image]:
    try:
        tracemalloc.start()
        t0 = time.perf_counter()

        img_sources = get_image_sources_from_url(url)
        max_workers = max_workers = min(32, (os.cpu_count() or 1) * 4)

        images = []

        executor = ThreadPoolExecutor(max_workers=max_workers)
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_image(session, src, idx, executor)
                for idx, src in enumerate(img_sources)
            ]
            results = await asyncio.gather(*tasks)
        executor.shutdown()

        images = [img for img in results if img is not None]

        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        

        metrics = {
            "url": url,
            "images_found": len(images),
            "time_sec": t1 - t0,
            "mem_current_kb": current / 1024,
            "mem_peak_kb": peak / 1024,
        }

        os.makedirs("output",exist_ok=True)

        with open("output/benchmark_report.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logging.logging.info(f"\n✅ Extracted {len(images)} images.")
        logging.logging.info(f"⏱️ Processing Time: {t1 - t0:.2f} seconds")
        logging.logging.info(f"💾 Memory Usage: Current = {current / 1024:.2f} KB, Peak = {peak / 1024:.2f} KB")
        logging.logging.info(f"Image Extraction Completed")

        return images
    except Exception as e:
        CustomExceptionHandling(e,sys)


# if __name__ == "__main__":
#     import sys
#     url = "https://en.wikipedia.org/wiki/Cancer" if len(sys.argv) < 2 else sys.argv[1]
#     asyncio.run(extract_images_from_website_async(url))
