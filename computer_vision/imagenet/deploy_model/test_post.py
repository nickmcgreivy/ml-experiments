import requests # type: ignore

url = 'http://127.0.0.1:8000'


files = {'imgfile': open('great_white_shark.JPEG', 'rb')}
k = 3
predict_url = url + f'/predict/?k={k}'

print(requests.post(predict_url, files=files).text)