
import wget
    
IMAGE_URLS = [
    'https://i.imgur.com/SdYYBDt.png',  # 0
    'https://i.imgur.com/Wy7mad6.png',  # 1
    'https://i.imgur.com/nhBZndj.png',  # 2
    'https://i.imgur.com/V6XeoWZ.png',  # 3
    'https://i.imgur.com/EdxBM1B.png',  # 4
    'https://i.imgur.com/zWSDIuV.png',  # 5
    'https://i.imgur.com/Y28rZho.png',  # 6
    'https://i.imgur.com/6qsCz2W.png',  # 7
    'https://i.imgur.com/BVorzCP.png',  # 8
    'https://i.imgur.com/vt5Edjb.png',  # 9
]



for url in IMAGE_URLS:
    wget.download(url)

