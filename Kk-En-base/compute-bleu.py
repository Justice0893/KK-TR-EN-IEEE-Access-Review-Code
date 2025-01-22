# Corpus BLEU - files must be untokenized/unsubworded
# Run this file from CMD/Terminal
# Example Command: python3 compute-bleu.py test_file_name.txt mt_file_name.txt


import sys
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

target_test = sys.argv[1]  # Test file argument
target_pred = sys.argv[2]  # Translated file argument

# Open the test dataset human translation file and detokenize the references
refs = []

with open(target_test) as test:
    for line in test: 
        line = line.strip()
        refs.append(line)

print("Reference first sentence:", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU


# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open(target_pred) as pred:  
    for line in pred: 
        line = line.strip()
        preds.append(line)

print("Translated first sentence:", preds[0])


bleu = BLEU()
print("BLEU: ", bleu.corpus_score(preds, refs))

chrf = CHRF()
print("CHRF: ",chrf.corpus_score(preds, refs))