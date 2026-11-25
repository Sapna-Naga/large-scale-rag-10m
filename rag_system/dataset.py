"""Download the original BEIR SciFact benchmark, not generated sample data."""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', default='data')
    a = p.parse_args()
    root = Path(a.directory)
    root.mkdir(parents=True,exist_ok=True)
    archive = root/'scifact.zip'
    with urllib.request.urlopen(URL,timeout=120) as response, archive.open('wb') as out:
        shutil.copyfileobj(response,out)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (root/name).resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive path')
        z.extractall(root)
    print(f'Extracted {archive} into {root}')

if __name__=='__main__':
    main()
