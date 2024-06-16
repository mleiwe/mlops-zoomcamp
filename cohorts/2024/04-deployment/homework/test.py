import requests

year_month = {
    "YEAR" : 2023,
    "MONT": 5
}

url = 'http://localhost:9696/predict' 
requests.post(url, json=year_month)