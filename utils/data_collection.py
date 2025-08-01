from icrawler.builtin import GoogleImageCrawler

def scrape_images(keyword, dir_path, max_num=120):
    crawler = GoogleImageCrawler(storage={'root_dir': dir_path})
    crawler.crawl(keyword=keyword, max_num=max_num)

categories = {
    "food photography": "Dataset/train/non_medical/food",
    "street scenes": "Dataset/train/non_medical/street",
    "wildlife animals": "Dataset/train/non_medical/wildlife",
    "fashion photography": "Dataset/train/non_medical/fashion",
    "architecture buildings": "Dataset/train/non_medical/buildings",
    "sports action": "Dataset/train/non_medical/sports",
    "vehicles on road": "Dataset/train/non_medical/vehicles",
    "underwater photos": "Dataset/train/non_medical/underwater",
    "mountain landscapes": "Dataset/train/non_medical/mountains",
    "abstract art": "Dataset/train/non_medical/abstract",
    "indoor home interiors": "Dataset/train/non_medical/interiors",
    "people walking on street": "Dataset/train/non_medical/people_street",
    "aerial drone shots": "Dataset/train/non_medical/aerial",
    "festival crowd": "Dataset/train/non_medical/crowd",
    "cute pets": "Dataset/train/non_medical/pets"
}

for keyword, path in categories.items():
    scrape_images(keyword, path, max_num=120)