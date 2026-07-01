FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY python_scripts/external_repositories/ python_scripts/external_repositories/
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY label_csvs/ label_csvs/
COPY stain_refs/ stain_refs/
COPY python_scripts/ python_scripts/

ENTRYPOINT ["python"]
CMD ["python_scripts/prepare_for_MIL/run_trident_stain_feats.py", "--wsi_dir", "data/raw_wsi/IgA", "--job_dir", "WSI/IgA/trident", "--labels_csv", "label_csvs/labels_unfiltered.csv", "--stain_refs_dir", "stain_refs/IgA_20x", "--backbone", "uni_v2", "--mag", "20", "--patch_size", "224", "--overlap", "0", "--batch_size", "256", "--search_nested"]