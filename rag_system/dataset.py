
"""
Download and validate the BEIR SciFact dataset.

Download sources:
    1. Hugging Face BEIR repository.
    2. Official BEIR ZIP archive.

The dataset is stored in the format required by the RAG system.
"""

import csv
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
import zipfile

from pathlib import Path


# Dataset configuration
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "scifact"

HF_DATASET = "BeIR/scifact"
HF_QRELS = "BeIR/scifact-qrels"

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)

# Checksum published by BEIR for the original ZIP archive.
BEIR_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"

MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 60

REQUIRED_FILES = [
    "corpus.jsonl",
    "queries.jsonl",
    "qrels/train.tsv",
    "qrels/test.tsv",
]


def read_jsonl_ids(file_path: Path) -> set:
    """Read and validate document or query IDs."""

    ids = set()

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            record_id = str(record["_id"]).strip()

            if not record_id:
                raise ValueError(
                    f"Empty ID in {file_path}, "
                    f"line {line_number}"
                )

            if record_id in ids:
                raise ValueError(
                    f"Duplicate ID in {file_path}: {record_id}"
                )

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"Invalid text in {file_path}, "
                    f"line {line_number}"
                )

            if not record["text"].strip():
                raise ValueError(
                    f"Empty text in {file_path}, "
                    f"line {line_number}"
                )

            ids.add(record_id)

    if not ids:
        raise ValueError(f"No records found in {file_path}")

    return ids


def validate_dataset(dataset_dir: Path) -> dict:
    """Check dataset files and their reference integrity."""

    for relative_path in REQUIRED_FILES:
        file_path = dataset_dir / relative_path

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset file missing: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise ValueError(f"Dataset file is empty: {file_path}")

    document_ids = read_jsonl_ids(
        dataset_dir / "corpus.jsonl"
    )

    query_ids = read_jsonl_ids(
        dataset_dir / "queries.jsonl"
    )

    statistics = {
        "documents": len(document_ids),
        "queries": len(query_ids),
    }

    # Verify that relevance judgments reference existing records.
    for split in ("train", "test"):
        qrels_path = dataset_dir / "qrels" / f"{split}.tsv"

        judgment_count = 0
        evaluated_queries = set()
        seen_pairs = set()

        with qrels_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file, delimiter="\t")

            required_columns = {
                "query-id",
                "corpus-id",
                "score",
            }

            if not required_columns.issubset(
                reader.fieldnames or []
            ):
                raise ValueError(
                    f"Invalid qrels columns: {qrels_path}"
                )

            for row in reader:
                query_id = str(row["query-id"])
                document_id = str(row["corpus-id"])

                if query_id not in query_ids:
                    raise ValueError(
                        f"Unknown query ID: {query_id}"
                    )

                if document_id not in document_ids:
                    raise ValueError(
                        f"Unknown document ID: {document_id}"
                    )

                pair = (query_id, document_id)

                if pair in seen_pairs:
                    raise ValueError(
                        f"Duplicate relevance judgment: {pair}"
                    )

                score = float(row["score"])

                if not score.is_integer() or score < 0:
                    raise ValueError(
                        f"Invalid relevance score: {score}"
                    )

                seen_pairs.add(pair)
                evaluated_queries.add(query_id)
                judgment_count += 1

        if judgment_count == 0:
            raise ValueError(
                f"No relevance judgments found: {qrels_path}"
            )

        statistics[f"{split}_judgments"] = judgment_count
        statistics[f"{split}_queries"] = len(
            evaluated_queries
        )

    return statistics


def write_jsonl(dataset, output_path: Path) -> None:
    """Write dataset records in BEIR JSONL format."""

    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            record = dict(row)

            # Preserve the original ID and document content.
            record["_id"] = str(record["_id"])

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def write_qrels(dataset, output_path: Path) -> None:
    """Write relevance judgments in BEIR TSV format."""

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(file, delimiter="\t")

        writer.writerow(
            ["query-id", "corpus-id", "score"]
        )

        for row in dataset:
            score = float(row["score"])

            if not score.is_integer() or score < 0:
                raise ValueError(
                    f"Invalid relevance score: {score}"
                )

            writer.writerow(
                [
                    str(row["query-id"]),
                    str(row["corpus-id"]),
                    int(score),
                ]
            )


def download_from_huggingface(target_dir: Path) -> None:
    """Download SciFact from the BEIR Hugging Face repository."""

    from datasets import load_dataset

    target_dir.mkdir(parents=True, exist_ok=True)

    qrels_dir = target_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)

    print("Downloading corpus from Hugging Face...")

    corpus = load_dataset(
        HF_DATASET,
        "corpus",
        split="corpus",
    )

    write_jsonl(
        corpus,
        target_dir / "corpus.jsonl",
    )

    print("Downloading queries from Hugging Face...")

    queries = load_dataset(
        HF_DATASET,
        "queries",
        split="queries",
    )

    write_jsonl(
        queries,
        target_dir / "queries.jsonl",
    )

    print("Downloading relevance judgments...")

    for split in ("train", "test"):
        qrels = load_dataset(
            HF_QRELS,
            split=split,
        )

        write_qrels(
            qrels,
            qrels_dir / f"{split}.tsv",
        )

    print("Hugging Face download completed.")


def calculate_md5(file_path: Path) -> str:
    """Calculate the checksum of the downloaded archive."""

    checksum = hashlib.md5(usedforsecurity=False)

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            checksum.update(block)

    return checksum.hexdigest()


def download_from_beir(target_dir: Path) -> None:
    """Download and validate the official BEIR ZIP archive."""

    target_dir.mkdir(parents=True, exist_ok=True)

    archive = target_dir / "scifact.zip.part"

    request = urllib.request.Request(
        BEIR_URL,
        headers={
            "User-Agent": "large-scale-rag-10m/1.0"
        },
    )

    print(f"Downloading from: {BEIR_URL}")

    with urllib.request.urlopen(
        request,
        timeout=DOWNLOAD_TIMEOUT,
    ) as response, archive.open("wb") as output:

        downloaded_bytes = 0

        while True:
            block = response.read(1024 * 1024)

            if not block:
                break

            output.write(block)
            downloaded_bytes += len(block)

            print(
                f"Downloaded "
                f"{downloaded_bytes / (1024 * 1024):.2f} MB",
                flush=True,
            )

    # Compare the archive with the official BEIR checksum.
    actual_md5 = calculate_md5(archive)

    if actual_md5 != BEIR_MD5:
        raise ValueError(
            "Archive checksum mismatch. "
            "The downloaded file may be incomplete or corrupted."
        )

    print("Archive checksum verified.")

    # Extract only the required dataset files.
    with zipfile.ZipFile(archive, "r") as zip_file:
        corrupted_file = zip_file.testzip()

        if corrupted_file is not None:
            raise ValueError(
                f"Corrupted archive member: {corrupted_file}"
            )

        for relative_path in REQUIRED_FILES:
            member = f"scifact/{relative_path}"

            output_path = target_dir / relative_path
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with zip_file.open(member) as source:
                with output_path.open("wb") as output:
                    shutil.copyfileobj(source, output)

    archive.unlink()

    print("Official BEIR download completed.")


def print_statistics(statistics: dict) -> None:
    """Display actual dataset counts."""

    print("\nDataset statistics")
    print("-" * 45)

    for name, value in statistics.items():
        label = name.replace("_", " ").capitalize()
        print(f"{label:<30}: {value:,}")

    print("-" * 45)


def main() -> None:
    """Download SciFact or reuse a valid local dataset."""

    print("=" * 55)
    print("BEIR SciFact Dataset Acquisition")
    print("=" * 55)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Do not download the dataset again if it is already valid.
    if DATASET_DIR.exists():
        try:
            statistics = validate_dataset(DATASET_DIR)

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            print(
                f"Existing dataset failed validation: {error}"
            )

        else:
            print("Existing dataset is valid.")
            print("Skipping download.")

            print_statistics(statistics)
            return

    # Try Hugging Face first, followed by the official archive.
    sources = [
        ("Hugging Face", download_from_huggingface),
        ("Official BEIR server", download_from_beir),
    ]

    errors = []

    for source_name, downloader in sources:

        for attempt in range(1, MAX_RETRIES + 1):

            print(
                f"\nSource: {source_name} | "
                f"Attempt {attempt}/{MAX_RETRIES}"
            )

            # Keep incomplete downloads outside the final directory.
            with tempfile.TemporaryDirectory(
                prefix="scifact_",
                dir=DATA_DIR,
            ) as temp_directory:

                temp_path = Path(temp_directory)

                try:
                    downloader(temp_path)

                    print("Validating downloaded dataset...")

                    statistics = validate_dataset(temp_path)

                    # Record the actual source and measured counts.
                    manifest = {
                        "dataset": "BEIR SciFact",
                        "source": source_name,
                        "statistics": statistics,
                    }

                    with (
                        temp_path / "manifest.json"
                    ).open("w", encoding="utf-8") as file:
                        json.dump(
                            manifest,
                            file,
                            indent=2,
                        )

                    # Replace an invalid local copy only after the
                    # newly downloaded dataset passes validation.
                    if DATASET_DIR.exists():
                        shutil.rmtree(DATASET_DIR)

                    shutil.move(
                        str(temp_path),
                        str(DATASET_DIR),
                    )

                    print("\nDataset downloaded successfully.")
                    print(f"Source: {source_name}")
                    print(
                        f"Location: {DATASET_DIR.resolve()}"
                    )

                    print_statistics(statistics)
                    return

                except Exception as error:
                    message = (
                        f"{source_name}, attempt {attempt}: "
                        f"{type(error).__name__}: {error}"
                    )

                    errors.append(message)
                    print(f"Download failed: {message}")

            if attempt < MAX_RETRIES:
                delay = 2 ** attempt

                print(
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

        print(
            f"All attempts failed for {source_name}. "
            "Trying the next available source."
        )

    print("\nAll dataset download sources failed.")

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()
