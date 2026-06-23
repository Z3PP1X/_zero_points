"""Script to calculate and add kappa value to each line in the specified CSV files."""

import csv
import json
import logging
from pathlib import Path


# Setup structured logging
class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Arguments:
            record: The LogRecord instance to format.

        Returns:
            The JSON string representation of the log.
        """
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent propagation to root logger to avoid double logging
logger.propagate = False


def find_column_indices(headers: list[str]) -> tuple[int, int, int]:
    """Finds the 0-indexed column positions for f'(x0), f''(x0), and kappa.

    Arguments:
        headers: List of CSV column headers.

    Returns:
        A tuple containing the indices of f'(x0), f''(x0), and kappa.

    Raises:
        ValueError: If f'(x0) or f''(x0) cannot be located in the headers.
    """
    f_prime_idx = -1
    f_double_prime_idx = -1
    kappa_idx = -1

    for i, h in enumerate(headers):
        # Normalize: strip quotes, whitespace, and case
        cleaned = h.strip().strip('"').strip("'").strip().lower()
        if cleaned in ["f'(x0)", "f'(x_0)", "d1x", "dfx", "f'(x)"]:
            f_prime_idx = i
        elif cleaned in ["f''(x0)", "f''(x_0)", "d2x", "d2fx", "f''(x)"]:
            f_double_prime_idx = i
        elif cleaned == "kappa":
            kappa_idx = i

    if f_prime_idx == -1:
        raise ValueError(f"Could not find f'(x0) column in headers: {headers}")
    if f_double_prime_idx == -1:
        raise ValueError(f"Could not find f''(x0) column in headers: {headers}")

    return f_prime_idx, f_double_prime_idx, kappa_idx


def process_csv_file(file_path: Path) -> None:
    """Processes a CSV file, adding or updating a 'kappa' column.

    Arguments:
        file_path: The Path object pointing to the CSV file to modify.

    Returns:
        None

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an I/O error occurs during file reading or writing.
        ValueError: If the required columns are missing or if file is invalid.
    """
    logger.info(f"Starting processing of file: {file_path}")

    # Read original contents with skipinitialspace=True to handle spaces before quotes
    rows: list[list[str]] = []
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            rows = list(reader)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}", exc_info=True)
        raise e
    except IOError as e:
        logger.error(f"Failed to read file: {file_path}", exc_info=True)
        raise e

    if not rows:
        raise ValueError(f"The file {file_path} is empty.")

    raw_header_row = rows[0]

    # Clean the headers by stripping outer quotes and whitespace
    header_row = [h.strip().strip('"').strip("'").strip() for h in raw_header_row]

    # Find the positions of f', f'', and optionally kappa
    try:
        f_prime_idx, f_double_prime_idx, kappa_idx = find_column_indices(header_row)
    except ValueError as e:
        logger.error(f"Error parsing headers for {file_path}: {e}", exc_info=True)
        raise e

    # Update header if kappa is new
    if kappa_idx == -1:
        new_header_row = header_row + ["kappa"]
    else:
        new_header_row = header_row

    new_rows: list[list[str]] = [new_header_row]

    # Process each data row
    for line_num, row in enumerate(rows[1:], start=2):
        if not row:
            continue

        # Check for matching columns count to avoid broken rows
        if len(row) < max(f_prime_idx, f_double_prime_idx) + 1:
            logger.warning(
                f"Row {line_num} in {file_path} has insufficient columns: {row}."
            )
            continue

        f_prime_str = row[f_prime_idx].strip().strip('"')
        f_double_prime_str = row[f_double_prime_idx].strip().strip('"')

        try:
            f_prime = float(f_prime_str)
            f_double_prime = float(f_double_prime_str)

            if f_prime == 0.0:
                kappa_val = 0
            else:
                raw_kappa = round(f_double_prime / (f_prime ** 2))
                kappa_val = max(-25, min(25, raw_kappa))

            kappa_str = str(kappa_val)
        except ValueError:
            logger.warning(
                f"Row {line_num} in {file_path} has non-float values "
                f"f'={f_prime_str}, f''={f_double_prime_str}."
            )
            kappa_str = ""

        # Construct/update the row
        if kappa_idx == -1:
            row_with_kappa = row + [kappa_str]
        else:
            # If row has fewer elements than kappa_idx, pad it
            row_updated = list(row)
            while len(row_updated) <= kappa_idx:
                row_updated.append("")
            row_updated[kappa_idx] = kappa_str
            row_with_kappa = row_updated

        new_rows.append(row_with_kappa)

    # Write back to file safely
    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(new_rows)
        logger.info(f"Successfully processed kappa in {file_path}")
    except IOError as e:
        logger.error(f"Failed to write updated CSV to {file_path}", exc_info=True)
        raise e


def main() -> None:
    """Main function to orchestrate processing of all datasets CSV files.

    Arguments:
        None

    Returns:
        None

    Raises:
        None
    """
    csv_files = [
        Path("datasets/run_20260603_123013/parallel_benchmark_results.csv"),
        Path("datasets/run_20260604_154509/parallel_benchmark_results.csv"),
        Path("datasets/run_20260604_154509/parallel_benchmark_results_orig.csv"),
        Path("datasets/derivatives_output.csv"),
    ]

    for file_path in csv_files:
        try:
            process_csv_file(file_path)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    main()
