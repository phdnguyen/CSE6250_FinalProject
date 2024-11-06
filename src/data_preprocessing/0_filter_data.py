from pathlib import Path
import pandas as pd

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

in_path = root_dir / 'data' / 'raw' / 'NOTEEVENTS.csv.gz'
out_path = root_dir / 'data' / 'processed' / '0_filtered_data.csv'

def main():

    notes = pd.read_csv(in_path, compression='gzip',
                        usecols=['ROW_ID','TEXT','CATEGORY'],
                        header=0,
                        sep=',',
                        quotechar='"'
                       )
    
    notes = notes[notes.CATEGORY=='Discharge summary'][['ROW_ID','TEXT']] #Use only discharge summary
    
    notes.to_csv(out_path,index=False)

if __name__ == "__main__":
    main()