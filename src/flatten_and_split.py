import json
import glob
import os
import logging
import random
from typing import Any, Dict, List, Tuple, Union

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# --------------------------- #
# Safe value replacement utils
# --------------------------- #
def safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""

def safe_int(value: Any) -> int:
    return int(value) if isinstance(value, int) else -1

def safe_float(value: Any) -> float:
    return float(value) if isinstance(value, (float, int)) else 0.0

def safe_list(value: Any) -> List:
    return value if isinstance(value, list) else []

def safe_dict(value: Any) -> Dict:
    return value if isinstance(value, dict) else {}


# --------------------------- #
# Flattening function
# --------------------------- #
def flatten_document(document: Dict[str, Any]) -> Dict[str, Any]:
    metadata = safe_dict(document.get("metadata"))
    authors = safe_list(metadata.get("authors"))
    fields_of_study = safe_list(metadata.get("fields_of_study"))
    license_info = safe_dict(metadata.get("license"))
    journal_info = safe_dict(metadata.get("journal"))

    return {
        "corpus_id": safe_int(document.get("corpus_id")),
        "doi": safe_str(metadata.get("doi")),
        "url": safe_str(metadata.get("url")),
        "title": safe_str(metadata.get("title")),
        "authors": [safe_str(a.get("name")) for a in authors],
        "author_ids": [safe_str(a.get("authorId")) for a in authors],
        "venue": safe_str(metadata.get("venue")),
        "year": safe_int(metadata.get("year")),
        "fields_of_study": [safe_str(f.get("category")) for f in fields_of_study],
        "publication_date": safe_str(metadata.get("publication_date")),
        "journal_name": safe_str(journal_info.get("name")),
        "license_publisher": safe_str(license_info.get("publisher")),
        "license": safe_str(license_info.get("license")),
        "oa_url": safe_str(license_info.get("oa_url")),
        "text": safe_str(document.get("text")),
        "field_classification": [safe_str(x) for x in safe_list(document.get("field_classification"))],
        "text_type": safe_str(document.get("text_type")),
        "embedding": [safe_float(x) for x in safe_list(document.get("embedding"))],
        "label": [safe_int(x) for x in safe_list(document.get("label"))],
    }


# --------------------------- #
# JSON file processing
# --------------------------- #
def process_json_files(input_dir: str) -> List[Dict[str, Any]]:
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    logging.info(f"Found {len(json_files)} JSON files in '{input_dir}'.")

    flattened_data = []
    failed_files = 0

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                flattened = flatten_document(raw_data)
                flattened_data.append(flattened)
        except Exception as e:
            logging.warning(f"Failed to process {file_path}: {e}")
            failed_files += 1

    logging.info(f"Processed {len(flattened_data)} files successfully.")
    if failed_files > 0:
        logging.warning(f"Skipped {failed_files} files due to errors.")

    return flattened_data


# --------------------------- #
# Dataset splitting
# --------------------------- #
def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Shuffle and split the dataset into train, validation, and test sets.
    """
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    logging.info(f"Split into: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")
    return train_data, val_data, test_data


# --------------------------- #
# Save output to JSONL
# --------------------------- #
def save_to_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info(f"Saved {len(data)} records to '{output_path}'")


# --------------------------- #
# Main entry point
# --------------------------- #
if __name__ == "__main__":
    INPUT_DIR = "entry_embeddings"
    OUTPUT_DIR = "output_splits"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Starting processing from: {INPUT_DIR}")
    flattened_records = process_json_files(INPUT_DIR)

    if flattened_records:
        train_data, val_data, test_data = split_dataset(flattened_records)

        save_to_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
        save_to_jsonl(val_data, os.path.join(OUTPUT_DIR, "validation.jsonl"))
        save_to_jsonl(test_data, os.path.join(OUTPUT_DIR, "test.jsonl"))
    else:
        logging.warning("No valid JSON files were processed.")
