## Data procurement

Experiments performed on [CRC](https://zenodo.org/record/1214456) and customised dataset from [CAMELYON17](https://camelyon17.grand-challenge.org/).


For the CAMELYON17 option, we adopt the same naming convention as in the [PatchCamelyon dataset](https://github.com/basveeling/pcam). However, the tiles are 256x256px extracted at 40x magnification. They are resized to 224x224px at train time. Download [CAMELYON17 dataset](https://registry.opendata.aws/camelyon/) using the [AWS command line interface](https://aws.amazon.com/cli/):

```
$ aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17 .
```

Run (or adapt) `camelyon_dataset.py` to produce train, validation, and test datasets from `meta_split.csv`. N.B. this requires a lot of RAM.
