# Data Directory

This directory should contain the skin cancer classification datasets.

## Expected Structure

```
data/
├── ham10000/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
├── isic2018/
│   ├── ISIC2018_Task3_Training_Input/
│   └── ISIC2018_Task3_Training_GroundTruth.csv
├── isic2019/
│   ├── ISIC_2019_Training_Input/
│   └── ISIC_2019_Training_GroundTruth.csv
└── isic2020/
    ├── train/
    └── ISIC_2020_Training_GroundTruth.csv
```

## Download Links

1. **HAM10000**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. **ISIC 2018**: https://challenge.isic-archive.com/data/#2018
3. **ISIC 2019**: https://challenge.isic-archive.com/data/#2019
4. **ISIC 2020**: https://challenge.isic-archive.com/data/#2020

## Note

Data files are git-ignored due to their size. Download and place them manually.
