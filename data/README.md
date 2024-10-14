# ISCO Job Classification - Data Preparation

PLACEHOLDER FOR DATA PREPARATION, INCLUDES:

- [x] Code for scraping and exporting ESCO data as json
- [] Code for creating synthetic job ads
- [x] ESCO data (json)
- [] Synthetic job ads
- [] ISCO data (labels only)
- [] Sub-sample of synthetic job ads (for "demo" purposes)

- Assumes you've:
    - downloaded the dataset [here](https://esco.ec.europa.eu/en/use-esco/download/privacy-statement?packages=v120_classification_en_csv/)
    - renamed it "ESCO_dataset"
    - saved it in the `data` subdirectory



## Requirements

- **Python**: 3.12 or higher

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Download and unzip [this dataset](https://esco.ec.europa.eu/en/use-esco/download/privacy-statement?packages=v120_classification_en_csv/), rename it "ESCO_dataset", and bring it into this `data` subdirectory

3. Run the main script:
    ```bash
    python get_data.py