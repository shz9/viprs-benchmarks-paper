import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/zhilizheng/SBayesRC/refs/heads/main/inst/extdata/ref4cM_v37.pos",
                 sep=r'\s+')

# Save it to a file:
df.to_csv("data/ldetect_data/EUR_blocks_4cM.pos", sep='\t', index=False)

df.drop(columns=['Block'], inplace=True)
df.columns = ['chr', 'start', 'stop']

# Convert chromosome to string and add 'chr' prefix:
df['chr'] = 'chr' + df['chr'].astype(str)

# Save to file:
df.to_csv("data/ldetect_data/EUR_blocks_4cM.bed", sep='\t', index=False)
