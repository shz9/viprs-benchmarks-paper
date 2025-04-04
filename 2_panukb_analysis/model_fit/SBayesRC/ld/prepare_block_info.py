import pandas as pd
import argparse


def process_bed_file(input_file, output_file):

    # Read the input file into a pandas DataFrame
    df = pd.read_csv(input_file, sep=r'\s+', header=0)

    # Add the "Block" column with enumerated IDs starting from 1
    df['Block'] = range(1, len(df) + 1)

    # Remove the "chr" from the Chromosome column
    df['chr'] = df['chr'].str.replace('chr', '', regex=False)

    # Rename columns according to the desired output
    df.rename(columns={'chr': 'Chrom', 'start': 'StartBP', 'stop': 'EndBP'}, inplace=True)

    # Reorder the columns to have "Block" first
    df = df[['Block', 'Chrom', 'StartBP', 'EndBP']]

    # Write the modified DataFrame to the output file
    df.to_csv(output_file, sep='\t', index=False)
    print(f"File processed and saved as {output_file}")


# Set up argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a BED file by adding Block column and modifying header.')
    parser.add_argument('-i', dest='input_file', help='Path to the input BED file')
    parser.add_argument('-o', dest='output_file', help='Path to save the processed output file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the input and output file paths
    process_bed_file(args.input_file, args.output_file)
