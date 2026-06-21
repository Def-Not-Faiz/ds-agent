import pathlib
import urllib.request

p = pathlib.Path('uploads/sample_test.csv')
p.parent.mkdir(exist_ok=True)
p.write_text(
    'age,city,income,bought\n'
    '25,Hyderabad,30000,yes\n'
    '30,Mumbai,40000,no\n'
    '40,Hyderabad,55000,yes\n'
    '45,Delhi,80000,no\n'
    '50,Mumbai,90000,yes\n', encoding='utf-8')

url = 'http://127.0.0.1:8000/analyze'
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
headers = {
    'Content-Type': f'multipart/form-data; boundary={boundary}'
}
body = []
body.append(f'--{boundary}')
body.append('Content-Disposition: form-data; name="prompt"')
body.append('')
body.append('predict whether a customer bought')
body.append(f'--{boundary}')
body.append('Content-Disposition: form-data; name="file"; filename="sample_test.csv"')
body.append('Content-Type: text/csv')
body.append('')
body.append(p.read_text())
body.append(f'--{boundary}--')
body.append('')
data = '\r\n'.join(body).encode('utf-8')
req = urllib.request.Request(url, data=data, headers=headers)
with urllib.request.urlopen(req, timeout=120) as resp:
    print('STATUS', resp.status)
    print(resp.read().decode('utf-8')[:2000])
