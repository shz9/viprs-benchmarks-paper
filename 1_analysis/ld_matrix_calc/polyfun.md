To obtain the statistics about the PolyFun LD matrices, we used the `aws` CLI client as follows:

1) First inspect the size of the repository:
```bash
aws s3 ls s3://broad-alkesgroup-ukbb-ld/UKBB_LD/ --recursive --human-readable --summarize --no-sign-request
```

We obtain the total size as: `3118639526669`, which translate to `3118.64 GB`.

2) Next we need to count the number of genetic variants represented in those matrices. To do this,
we download the metadata files:

```bash
aws s3 cp s3://broad-alkesgroup-ukbb-ld/UKBB_LD/ ~/Downloads/ukbb_ld/ --recursive --exclude "*" --include "chr*.gz" --no-sign-request
```

Then parse them and count the number of unique `rsid`s with python 
(this is important because regions are **overlapping**):

```python
import pandas as pd
from tqdm import tqdm
import glob

def get_unique_values():
    unique_values = set()
    
    files = glob.glob("/Users/szabad/Downloads/ukbb_ld/*.gz")
    
    for file in tqdm(files, total=len(files)): 
        df = pd.read_csv(file, sep=r'\s+')
        unique_values.update(set(list((df['rsid'] + df['allele1'] + df['allele2']).unique())))
    
    return unique_values

unique_snps = get_unique_values()
print(len(unique_snps))

```

We obtain the number of unique genetic variants as: `19465724`.