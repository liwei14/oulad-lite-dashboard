# Raw OULAD tables (deploy bundle)

Six small CSVs are **committed** for Streamlit Cloud (`courses`, `studentInfo`, `studentRegistration`, `assessments`, `studentAssessment`, `vle`).

`studentVle.csv` (~433 MB) is **not** in git. The app uses pre-built `data/interim/clickstream_bbb_ddd_d28.parquet` instead.

To rebuild features locally, symlink or copy `studentVle.csv` from the [OULAD download](https://analyse.kmi.open.ac.uk/open_dataset/download), then run the pipeline in the project README.
