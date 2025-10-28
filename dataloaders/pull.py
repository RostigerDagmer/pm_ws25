import asyncio
import aiohttp
import yaml
import os
import re
import argparse
import gzip
import zipfile
import shutil
from aiohttp import ClientTimeout


# ---------------- Helpers ----------------


def sanitize_name(name: str) -> str:
    """Turn a DOI/UUID or URL into a filesystem-safe folder name."""
    name = name.strip().replace("https://", "").replace("http://", "")
    name = re.findall(
        "(?:datasets/|uuid:|/dataset/[^/]+/)([0-9a-fA-F-]+|\d+)(?=$|[/?#])",  # noqa: W605
        name,
    )[0]
    return name


def decompress_gzip(filepath: str, delete_original=False):
    """Decompress .gz file and optionally delete original."""
    if not filepath.endswith(".gz"):
        return

    target_path = filepath[:-3]
    if os.path.exists(target_path):
        print(
            f"[=] Skipped decompression (exists): {os.path.basename(target_path)}"
        )
        return

    try:
        with (
            gzip.open(filepath, "rb") as f_in,
            open(target_path, "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        print(
            f"[↓] Decompressed {os.path.basename(filepath)} → {os.path.basename(target_path)}"
        )
        if delete_original:
            os.remove(filepath)
    except Exception as e:
        print(f"[!] Failed to decompress {filepath}: {e}")


def decompress_zip(filepath: str, delete_original: bool = False):
    """Decompress a .zip archive into the same directory as the file.

    Extracts all members. If the archive was already extracted before,
    existing files are left untouched. Optionally deletes the .zip after.
    """
    if not filepath.endswith(".zip"):
        return

    target_dir = os.path.dirname(filepath)
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            for member in zf.namelist():
                target_path = os.path.join(target_dir, member)
                if os.path.exists(target_path):
                    print(f"[=] Skipped (exists): {member}")
                    continue
                zf.extract(member, target_dir)
                print(f"[↓] Extracted {member}")
        if delete_original:
            os.remove(filepath)
    except Exception as e:
        print(f"[!] Failed to unzip {filepath}: {e}")


def extract_filename_from_cd(cd_header: str) -> str | None:
    """Extract filename from Content-Disposition header."""
    if not cd_header:
        return None
    match = re.search(
        r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
        cd_header,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


async def fetch_file(session, url, output_dir, index, delete_gz=False):
    """Download a single file asynchronously, relying on server filename."""
    filename = f"file_{index}.dat"  # fallback if server doesn’t send filename
    filepath = os.path.join(output_dir, filename)

    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"[!] Failed: {url} (status {resp.status})")
                return

            cd = resp.headers.get("Content-Disposition")
            real_name = extract_filename_from_cd(cd)
            if real_name:
                filename = real_name
                filepath = os.path.join(output_dir, filename)

            # Skip if already exists
            if os.path.exists(filepath):
                print(f"[=] Skipped (exists): {filename}")
                if filename.endswith(".gz"):
                    decompress_gzip(filepath, delete_original=delete_gz)
                elif filename.endswith(".zip"):
                    decompress_zip(filepath, delete_original=delete_gz)
                return

            content = await resp.read()
            with open(filepath, "wb") as f:
                f.write(content)

            print(f"[✓] Downloaded {filename}")
            if filename.endswith(".gz"):
                decompress_gzip(filepath, delete_original=delete_gz)
            elif filename.endswith(".zip"):
                decompress_zip(filepath, delete_original=delete_gz)

    except Exception as e:
        print(f"[!] Error downloading {url}: {e}")


async def download_all(sources_file, output_dir, delete_gz=False):
    """Main function to read YAML and download all files asynchronously."""
    with open(sources_file, "r") as f:
        data = yaml.safe_load(f)

    sources = data.get("sources", [])
    if not sources:
        print(f"No 'sources' found in {sources_file}.")
        return

    sources = list({s['source']: s for s in sources}.values())  # dedup
    existing = set(os.listdir(output_dir))
    sources = list(
        filter(
            lambda src: sanitize_name(src.get("source")) not in existing,
            sources,
        )
    )
    uuids = list(map(sanitize_name, [s.get("source") for s in sources]))

    print(f"{len(existing)} existing.")
    print(f"{len(uuids)} new.")

    timeout = ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit_per_host=5, ssl=False)

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector
    ) as session:
        tasks = []
        for src in sources:
            source_url = src.get("source")
            files = src.get("files", [])
            if not files:
                print(f"[!] No files for source: {source_url}")
                continue

            src_name = sanitize_name(source_url)
            src_dir = os.path.join(output_dir, src_name)

            if os.path.exists(src_dir):
                continue

            os.makedirs(src_dir, exist_ok=True)

            for i, file_url in enumerate(files, 1):
                tasks.append(
                    fetch_file(session, file_url, src_dir, i, delete_gz)
                )

        await asyncio.gather(*tasks)


# ---------------- CLI interface ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Asynchronously download all files from sources listed in a YAML file."
    )
    parser.add_argument(
        "--sources",
        "-s",
        default="./dataloaders/sources.yaml",
        help="Path to YAML file containing list of URLs under 'sources:'",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data",
        help="Directory to store downloaded files (created if missing)",
    )
    parser.add_argument(
        "--auto_delete",
        "-d",
        default=True,
        help="Automatically deletes archive files after they are decompressed.",
    )
    args = parser.parse_args()

    asyncio.run(
        download_all(args.sources, args.output, delete_gz=args.auto_delete)
    )


if __name__ == "__main__":
    main()
