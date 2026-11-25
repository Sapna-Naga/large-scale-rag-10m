"""Download the original BEIR archive, verify its published MD5, extract safely."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def main():
    root = Path("data").resolve()
    root.mkdir(exist_ok=True)
    archive = root / "scifact.zip"
    urllib.request.urlretrieve(URL, archive)
    with archive.open("rb") as f:
        actual = hashlib.file_digest(f, "md5").hexdigest()
    if actual != MD5:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe archive path")
        z.extractall(root)
    print(root / "scifact")


if __name__ == "__main__":
    main()
