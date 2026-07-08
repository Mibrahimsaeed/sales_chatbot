from etl.extract import extract_all
from etl.transform import transform
from etl.load import load_all

raw = extract_all()
data = transform(raw)

print(load_all(data))