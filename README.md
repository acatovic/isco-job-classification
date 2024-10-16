# isco-job-classification
Classification pipeline that categorizes online job postings into standardized ISCO categories - based on the EU Stats competition

## Requirements

- **Python**: 3.12 or higher
- It's preferrable to use a virtual environment such as `venv` to manage dependencies

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the pipeline:
    ```bash
    python pipeline/run.py --data <path/to/job_ads.csv> --occupations <path/to/occupations.json> --output <path/to/output_directory>
    ```

## Example

```bash
python pipeline/run.py --data data/sample_job_postings.csv --occupations data/occupation_data.json --output output/
```

