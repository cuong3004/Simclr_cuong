import gdown

data_root = 'data/CelebA'

url_celebA = 'https://drive.google.com/uc?id=1L-SEfeUNmYygYP30Cr8EuhNcyHwOCEnW'

download_path = f'{data_root}/img_align_celeba.zip'

if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(dataset_folder)

gdown.download(url, download_path, quiet=False)

with zipfile.ZipFile(download_path, 'r') as ziphandler:
    ziphandler.extractall(dataset_folder)
