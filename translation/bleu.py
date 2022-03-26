import sacrebleu
from sacremoses import MosesDetokenizer
import tensorflow_text as tf_text
import unidecode

md = MosesDetokenizer(lang='swe')

# Open the test dataset human translation file and detokenize the references
refs = []

with open("ref.txt") as test:
    for line in test:
        line = unidecode.unidecode(line)
        line = line.lower()
        line = line.strip().split()
        line = md.detokenize(line)

        refs.append(line)

print("Reference 1st sentence:", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU

# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open("pred.txt") as pred:
    for line in pred:
        line = line.strip().split()
        line = md.detokenize(line)
        preds.append(line)

print("MTed 1st sentence:", preds[0])

# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds,refs)
print(bleu.score)