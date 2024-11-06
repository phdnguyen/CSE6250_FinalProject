from pathlib import Path
import pandas as pd
import re
import nltk

#nltk.download('punkt_tab') #uncomment to download this package

#nltk.download('stopwords') #uncomment to download this package

from nltk.corpus import stopwords

# Set the root directory to the project root
root_dir = Path(__file__).resolve().parents[2]

in_path = root_dir / 'data' / 'processed' / '0_filtered_data.csv'
out_path = root_dir / 'data' / 'processed' / '1_cleaned_data.csv'
label_path = root_dir / 'data' / 'external' / 'NIHMS1767978-supplement-MIMIC_SBDH.csv'
contractions_path = root_dir / 'data' / 'external' / 'contractions.json'


contractions = pd.read_json(contractions_path, typ='series') #getting the list of English contractions
contractions = contractions.to_dict()


c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')


def clean_text(text):
    
    text = str(text)
    text = text.lower()
    text = BAD_SYMBOLS_RE.sub(' ', text)
    
    #expand contraction
    text = expandContractions(text)

    #remove punctuation
    text = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split())

    #remove numbers

    text = ' '.join([word for word in text.split() if not word.isnumeric()])

    #stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    text = ' '.join(filtered_sentence)

        
    return text

def main():

    # put in label
    
    label_source = pd.read_csv(label_path,usecols=['row_id','behavior_tobacco'])

    notes = pd.read_csv(in_path)

    labeled_notes = label_source.merge(notes,
                                       how='left',
                                       left_on = ['row_id'],
                                       right_on = ['ROW_ID']   
                                      )

    # make another column for binary labels (i.e. only smoking vs. non smoking)

    labeled_notes['behavior_tobacco_binary'] = labeled_notes['behavior_tobacco'].apply(lambda x: 0 if x in [0,1] else 1)

    # clean the TEXT column 
    
    labeled_notes['TEXT'] = labeled_notes['TEXT'].apply(clean_text)

    # save
    
    labeled_notes[['TEXT','behavior_tobacco','behavior_tobacco_binary']].to_csv(out_path,index=False)

if __name__ == "__main__":
    main()
