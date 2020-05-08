import requests
import os

url_base = 'https://github.com/asxgym/asx_data_daily/raw/master/data/'


def create_directory_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_file(name):
    url = f'{url_base}{name}'
    r = requests.get(url)
    with open(f'data/{name}', 'wb') as f:
        f.write(r.content)
