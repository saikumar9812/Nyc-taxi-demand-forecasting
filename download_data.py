"""Download the NYC TLC yellow-taxi CSVs (with pickup/dropoff coordinates) used by the notebook into ./data."""
import os
import sys
import urllib.request

# Mirror of the original TLC CSVs; the official TLC site now only serves parquet without lat/lon columns.
BASE_URL = "https://storage.googleapis.com/anaconda-public-data/nyc-taxi/csv"
FILES = {
    "yellow_tripdata_2015-01.csv": ("2015", 1985964692),
    "yellow_tripdata_2016-01.csv": ("2016", 1708674492),
    "yellow_tripdata_2016-02.csv": ("2016", 1783554554),
    "yellow_tripdata_2016-03.csv": ("2016", 1914669757),
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download(name, year, expected_size):
    dest = os.path.join(DATA_DIR, name)
    if os.path.exists(dest) and os.path.getsize(dest) == expected_size:
        print(f"[skip] {name} already downloaded")
        return
    url = f"{BASE_URL}/{year}/{name}"
    tmp = dest + ".part"
    print(f"[get ] {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        done = 0
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r       {done / 1e9:.2f} / {expected_size / 1e9:.2f} GB")
            sys.stdout.flush()
    print()
    if os.path.getsize(tmp) != expected_size:
        raise RuntimeError(f"{name}: size mismatch ({os.path.getsize(tmp)} != {expected_size})")
    os.replace(tmp, dest)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    wanted = sys.argv[1:] or list(FILES)
    for name in wanted:
        year, size = FILES[name]
        download(name, year, size)
    print("Done.")
