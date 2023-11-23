import evaluate
import sys
import numpy as np
from datasets import load_dataset, Sequence,ClassLabel
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import os 

os.environ['WANDB_DISABLED'] = 'true'


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    #import pdb; pdb.set_trace()
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



id2label = {
    0: "O",
    1: "1",
}
label2id = {
    "O": 0,
    "1": 1,
}






from datasets import Dataset
from get_data_dict import get_data_dict


model_name = "distilbert-base-uncased" # 0.653
model_name = 'xlm-roberta-base' 

# te 2 są najlepsze
model_name = 'xlm-roberta-large' # 0.668
model_name= 'microsoft/mdeberta-v3-base' # 0.68
model_name= 'l3cube-pune/malayalam-bert' # 0.666
model_name= 'l3cube-pune/kannada-bert' # 0.673
model_name = 'Twitter/twhin-bert-large' # 0.683

model_name=sys.argv[1]
num_epochs=2


tokens_list, labels_list = get_data_dict()
ds = Dataset.from_dict({"tokens": [['a','b','c'], ['a','e']], "ner_tags": [['0','0','1'], ['1','0']]})
ds = Dataset.from_dict({"tokens": tokens_list, "ner_tags": labels_list})
ner_tags_class_label = Sequence(ClassLabel(num_classes=2, names = ['0','1']))
ds  = ds.cast_column('ner_tags', ner_tags_class_label)
ds = ds.train_test_split(seed=123)
wnut = ds

#label_list = wnut["train"].features[f"ner_tags"].feature.names
label_list = ['0','1']
label_list = ['O','B']


tokenizer = AutoTokenizer.from_pretrained(model_name)

example = wnut["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])


tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")


labels = [label_list[i] for i in example[f"ner_tags"]]

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id
)

model_save_path = 'dravidian_model_' + model_name
training_args = TrainingArguments(
    output_dir=model_save_path,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=None,
    metric_for_best_model='eval_f1',
    warmup_ratio=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model()

print(model_name)
print('final model eval f1', trainer.evaluate(trainer.eval_dataset)['eval_f1'])


text = 'ಬ್ರೋ ಅವರು ಗೆ ದೇಶದ ಬಗ್ಗೆ ಅಭಿಮಾನ ಇಲ್ಲಾ ಬಿಡಿ'
text = 'ದೇಶಧ್ರೋಹಿಗಳು ಡಿಸ್ ಲೈಕ್ ಮಾಡಿದರೆ ನೀವು ನಿಜವಾದ ದೇಶ ದ್ರೋಹಿಗಳು ನಾಚಿಕೆ ಆಗಬೇಕು ನಿಮ್ಮ ಜನ್ಮಕ್ಕೆ.... ಇಂತಹ ಅದ್ಭುತವಾದ ವಿಡಿಯೋ ಗಳಿಗೂ ಡಿಸ್ ಲೈಕ್  ಮಾಡಿದ್ರಲ್ಲ ನಿಮಗೆ ನಾಚಿಕೆ ಆಗಲ್ವಾ.....:'
classifier = pipeline("ner", model=f"./{model_save_path}/")

out = classifier(text)
#print(out)

with open('../data/kannada_test_EACL24.csv', encoding='utf-8-sig') as f:
    lines = [x.rstrip('\n') for x in  f.readlines()]

out_lines = list()
for line in lines:
    out = classifier(line)
    try:
        start = out[0]['start']
        end = out[-1]['end']
        l = [i for i in range(start, end)]
    except IndexError:
        l = []
    l = str(l).replace(" ","")
    out_lines.append(l)

with open('kubapok_span_supervised.csv','w', encoding='utf-8-sig') as f:
    f.write('Text,Span\n')
    for line, out_line in zip(lines, out_lines):
        f.write(line+','+'"'+out_line+'"\n')
