"""
download_data.py
Downloads the UCI Individual Household Electric Power Consumption dataset,
extracts household_power_consumption.txt to data/raw/, and verifies the download.
"""

import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
TARGET_FILENAME = "household_power_consumption.txt"


def get_project_root() -> Path:
    """Returns the project root directory (household_energy_forecaster)."""
    return Path(__file__).resolve().parent.parent


def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file from a URL to dest_path with progress indication."""
    print(f"Downloading dataset from:\n  {url}\nTo:\n  {dest_path}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks
        downloaded = 0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent:6.2f}% ({mb_downloaded:.2f} MB / {mb_total:.2f} MB)")
                sys.stdout.flush()
            else:
                mb_downloaded = downloaded / (1024 * 1024)
                sys.stdout.write(f"\rDownloaded: {mb_downloaded:.2f} MB")
                sys.stdout.flush()

    print("\nDownload complete.")


def extract_file(zip_path: Path, target_filename: str, extract_dir: Path) -> Path:
    """Extracts target_filename from zip_path into extract_dir."""
    print(f"\nExtracting '{target_filename}' from '{zip_path.name}' to '{extract_dir}'...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    target_path = extract_dir / target_filename

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        namelist = zip_ref.namelist()
        matching_names = [name for name in namelist if Path(name).name == target_filename]
        if not matching_names:
            raise FileNotFoundError(f"'{target_filename}' not found in archive. Available: {namelist}")

        archive_member = matching_names[0]
        with zip_ref.open(archive_member) as source, open(target_path, "wb") as target:
            shutil.copyfileobj(source, target)

    print(f"Extraction complete: {target_path}")
    return target_path


def verify_dataset(file_path: Path) -> bool:
    """Verifies that the dataset file exists, has expected size and content."""
    print(f"\nVerifying dataset file at: {file_path}")
    if not file_path.exists():
        print("ERROR: Extracted file does not exist.")
        return False

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"File size: {size_bytes:,} bytes (~{size_mb:.2f} MB)")

    if size_bytes < 10 * 1024 * 1024:
        print("WARNING: Extracted file is unexpectedly small (< 10 MB).")
        return False

    # Read the first few lines to inspect header and sample rows
    print("\nFirst 5 lines of the dataset:")
    print("-" * 60)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx in range(5):
            line = f.readline()
            if not line:
                break
            print(f"Line {idx + 1}: {line.strip()}")
    print("-" * 60)

    print("\nVerification SUCCESSFUL: Dataset is ready for processing.")
    return True


def main() -> None:
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "individual_household_electric_power_consumption.zip"
    target_path = raw_dir / TARGET_FILENAME

    # Download zip if neither zip nor extracted file exists
    if not target_path.exists():
        if not zip_path.exists():
            download_file(DATA_URL, zip_path)
        else:
            print(f"Archive already exists at: {zip_path}")
        extract_file(zip_path, TARGET_FILENAME, raw_dir)
    else:
        print(f"Target file already exists: {target_path}")

    # Verify extracted data
    is_valid = verify_dataset(target_path)
    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
