import requests

url = "http://localhost:5000/api/carcut/url/"

payload={'Url': 'https://storage.googleapis.com/car_cut/data/1.jpeg'}
files=[

]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

# print(response._content)
file = open("test_result.png", "wb")
file.write(response.content)
file.close()
