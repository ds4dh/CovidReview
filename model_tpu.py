'''
Code for training a given model in each of the 5 folds, the test set is then predicted for the best evaluation f1-score epoch

Has been used that way:

source my_env/bin/activate
for value in digitalepidemiologylab/covid-twitter-bert-v2 dmis-lab/biobert-v1.1 roberta-base roberta-large microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
do
    python /data/user/knafou/Projects/RiskLick/COVID_classification/model_tpu.py $value
done

'''

import os, sys, logging
os.environ['TPU_IP_ADDRESS'] = '10.240.1.2'
os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;' + os.environ['TPU_IP_ADDRESS'] + ':8470'
os.environ['XLA_USE_BF16'] = '0'

import torch, random, numpy as np, time, glob
from torch.utils.data import DataLoader, IterableDataset
from sklearn.metrics import classification_report
import torch_xla
import torch_xla.distributed.parallel_loader as pl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
logging.basicConfig(level=logging.ERROR)
import tensorflow as tf
import nltk
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

model_name = sys.argv[1]

PREPATH = '/data/user/knafou'
PATH_TO_PROJECT = PREPATH + '/Projects/RiskLick/COVID_classification/'
SLIDING_WINDOW = False

DATA_PATH = PATH_TO_PROJECT + 'dataset/'
MODEL_PATH = '/data/user/knafou/RiskLick_COVID/'
if SLIDING_WINDOW:
    MODEL_PATH = '/data/user/knafou/RiskLick_COVID_WSW/'

MODEL_DIR = MODEL_PATH + model_name.replace('/', '-') + '/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
SPLITS = ['train', 'dev', 'test']

BATCH_SIZE = 6
LEARNING_RATE = 3e-5
EPOCHS = 20
N_CHECKPOINT = EPOCHS * 2
N_PROCESS = 8
WARMUP_PROPORTION = 0.1
N_FOLD = 5
label2index = {i: j for j, i in enumerate(["EPI: Case report",
              "EPI: Case series",
              "EPI: Case-control study",
              "EPI: Cohort study",
              "EPI: Cross-sectional study",
              "EPI: Diagnostic study",
              "EPI: Ecological study",
              "EPI: Guidelines",
              "EPI: Modelling study",
              "EPI: Other",
              "EPI: Outbreak or surveillance report",
              "EPI: Qualitative study",
              "EPI: Review",
              "EPI: Trial",
              "BASIC: Animal experiment",
              "BASIC: In vitro experiment",
              "BASIC: Biochemical/protein structure studies",
              "BASIC: Sequencing and phylogenetics",
              "BASIC: Within-host modelling",
              "BASIC: Basic research review",
              "Other",
              "Comment, editorial, ..., non-original"])}

index2label = {v: k for k, v in label2index.items()}
NUM_LABELS = len(label2index)

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 512

def main():

    for k_fold in range(2, N_FOLD):
        SIZE, line_number2doc_id = create_tf_records(path_to_data=DATA_PATH, splits=SPLITS, k_fold=k_fold)
        total_steps = {split: int(SIZE[split] / BATCH_SIZE * EPOCHS / N_PROCESS) for split in SPLITS}
        checkpoint_step = int(total_steps['train'] / N_CHECKPOINT)
        flags = {'total_steps': total_steps,
                 'checkpoint_step': [step for step in range(total_steps['train']) if step % checkpoint_step == 0][1:-1] + [total_steps['train']],
                 'line_number2doc_id': line_number2doc_id,
                 'model_dir': MODEL_DIR + 'fold=' + str(k_fold) + '/',
                 'k_fold': k_fold}

        for split in SPLITS:
            flags['split'] = split
            if split != 'train':

                if flags['split'] == 'test':
                    model_paths = [flags['model_dir'] + "torch_model_" + best_model_step]
                    flags['model_path'] = model_paths[0]
                    xmp.spawn(eval, args=(flags,), nprocs=N_PROCESS, start_method='fork')

                best_model_step = predictions2results(flags)

            else:
                xmp.spawn(train, args=(flags,), nprocs=N_PROCESS, start_method='fork')

def predictions2results(flags):

    predictions_path = flags['model_dir'] + flags['split'] + '_predictions/'
    results_path = flags['model_dir'] + flags['split'] + '_results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    predictions_paths = list(glob.glob(predictions_path + 'torch_model_*'))

    model = {}
    for predictions_path in predictions_paths:
        model_step = predictions_path.split('_')[-2]
        if model_step not in model:
            model[model_step] = {}
        with open(predictions_path) as f:
            predictions_file = f.read().strip()
        for line in predictions_file.split('\n'):
            id, label, *predictions = line.split('\t')
            y_pred = np.array([float(p) for p in predictions])
            if id in model[model_step]:
                model[model_step][id]['prediction'] += y_pred
            else:
                model[model_step][id] = \
                    {'prediction': y_pred,
                     'label': label}

    best_f1 = 0
    best_model_step = ''
    for model_step in model.keys():
        labels = []
        predictions = []
        for id in model[model_step].keys():
            label_index = np.argmax(model[model_step][id]['prediction'])
            labels.append(model[model_step][id]['label'])
            predictions.append(index2label[label_index])

        report = classification_report(labels, predictions, output_dict=True, digits=6)
        report_text = classification_report(labels, predictions, digits=4)

        with open(results_path + 'torch_model_' + model_step, 'w') as f:
            f.write(report_text)

        if report['weighted avg']['f1-score'] > best_f1:
            best_f1 = report['weighted avg']['f1-score']
            best_model_step = model_step

    if flags['split'] == 'test':
        logging.error(report_text)
        return None

    elif flags['split'] == 'dev':
        return best_model_step

def train(xla_process_idx, flags):

    device = xm.xla_device()
    seed_val = 1234
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    predictions_path = flags['model_dir'] + 'dev_predictions/'
    if xla_process_idx == 0:
        if not os.path.exists(flags['model_dir']):
            os.mkdir(flags['model_dir'])
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)


    train_streamer = DatasetStreamer(DATA_PATH + 'dataset_train_fold=' + str(flags['k_fold']) + '.tfrecord',
                                     split='train', seed=10*xla_process_idx)
    train_loader = DataLoader(train_streamer, collate_fn=pytorch_convert, batch_size=BATCH_SIZE)
    para_train_loader = pl.ParallelLoader(train_loader, [device], fixed_batch_size=True).per_device_loader(device)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS).train().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(flags['total_steps']['train'] * WARMUP_PROPORTION),
                                                num_training_steps=flags['total_steps']['train'])
    model.zero_grad()
    t = tqdm(total=flags['total_steps']['train'])
    from_time = time.time()
    step = 0
    total_loss = []

    for batch in para_train_loader:
        output = model(batch[0], token_type_ids=None, attention_mask=batch[1],
                     labels=batch[2])
        total_loss.append(output['loss'].item())
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        xm.optimizer_step(optimizer, barrier=True)
        scheduler.step()
        model.zero_grad()
        t.update(1)
        step += 1
        t.set_description('[training...] Process=' + str(xla_process_idx) +
                          ' train_avg_loss=' + "{:2.4f}".format(np.mean(total_loss[-(flags['checkpoint_step'][1]-flags['checkpoint_step'][0]):])))
        if step in flags['checkpoint_step'] :
            # computing dev
            t.set_description('[computing dev...] Process=' + str(xla_process_idx) +
                              ' train_avg_loss=' + "{:2.4f}".format(
                np.mean(total_loss[-(flags['checkpoint_step'][1] - flags['checkpoint_step'][0]):])))
            ids = []
            labels = []
            predictions = []
            dev_streamer = DatasetStreamer(DATA_PATH + 'dataset_dev_fold=' + str(flags['k_fold']) + '.tfrecord', split='eval',
                                       index=xla_process_idx)
            dev_loader = DataLoader(dev_streamer, collate_fn=pytorch_convert, batch_size=BATCH_SIZE)
            para_dev_loader = pl.ParallelLoader(dev_loader, [device], fixed_batch_size=False).per_device_loader(device)
            model = model.eval()
            for batch in para_dev_loader:
                with torch.no_grad():
                    outputs = model(batch[0], token_type_ids=None, attention_mask=batch[1])
                    labels.append([index2label[l] for l in batch[2].to('cpu').numpy()])
                    predictions.append(['\t'.join([str(p) for p in o]) for o in outputs[0].to('cpu').numpy()])
                    ids.append([flags['line_number2doc_id']['dev'][b] for b in batch[3].to('cpu').numpy()])
            model = model.train()
            labels = [item for sublist in labels for item in sublist]
            predictions = [item for sublist in predictions for item in sublist]
            ids = [item for sublist in ids for item in sublist]
            txt = ''
            for id, label, prediction in zip(ids, labels, predictions):
                txt += id + '\t' + label + '\t' + prediction + '\n'
            with open(predictions_path + 'torch_model_' + str(step) + '_' + str(xla_process_idx), 'w') as f:
                f.write(txt)
            xm.save(model.state_dict(), flags['model_dir'] + '/torch_model_'+ str(step))

        if step == flags['total_steps']['train'] :
            break

    to_time = time.time()
    logging.warning('Process ' + str(xla_process_idx) + '; time for training = ' +
                    "{:2.4f}".format((to_time - from_time) / 60) + ' minutes.')

def eval(xla_process_idx, flags):

    device = xm.xla_device()
    predictions_path = flags['model_dir'] + flags['split'] + '_predictions/'
    if not os.path.exists(predictions_path) and xla_process_idx == 0:
        os.mkdir(predictions_path)

    ids = []
    labels = []
    predictions = []
    streamer = DatasetStreamer(DATA_PATH + 'dataset_' + flags['split'] + '_fold=' + str(flags['k_fold']) + '.tfrecord', split='eval',
                               index=xla_process_idx)
    loader = DataLoader(streamer, collate_fn=pytorch_convert, batch_size=BATCH_SIZE-1)
    para_loader = pl.ParallelLoader(loader, [device], fixed_batch_size=False).per_device_loader(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS).to(device)
    ckpt = xser.load(flags['model_path'])
    model.load_state_dict(ckpt)
    model = model.eval()

    for batch in para_loader:
        with torch.no_grad():
            outputs = model(batch[0], token_type_ids=None, attention_mask=batch[1])
            labels.append([index2label[l] for l in batch[2].to('cpu').numpy()])
            predictions.append(['\t'.join([str(p) for p in o]) for o in outputs[0].to('cpu').numpy()])

            ids.append([flags['line_number2doc_id'][flags['split']][b] for b in batch[3].to('cpu').numpy()])

    labels = [item for sublist in labels for item in sublist]
    predictions = [str(item) for sublist in predictions for item in sublist]
    ids = [item for sublist in ids for item in sublist]

    logging.warning('Process ' + str(xla_process_idx) + ': writing prediction file')
    txt = ''
    for id, label, prediction in zip(ids, labels, predictions):
        txt += id + '\t' + label + '\t' + prediction + '\n'
    with open(predictions_path + flags['model_path'].split('/')[-1] + '_' + str(xla_process_idx), 'w') as f:
        f.write(txt)

class DatasetStreamer(IterableDataset):

    def __init__(self, file_path, split='train', seed=None, index=None):
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset([file_path])
        self.split = split
        if self.split == 'eval':
            dataset = dataset.shard(num_shards=N_PROCESS, index=index)
        dataset = dataset.map(extract_fn, num_parallel_calls=4)
        if self.split == 'train':
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=5000, seed=seed)
        dataset = dataset.prefetch(BATCH_SIZE * 10)
        self.dataset = dataset.batch(1)

    def parse_file(self):
        for batch in self.dataset:
            batch['title_ids'] = tokenizer.encode(batch['title_ids'].numpy()[0].decode('utf-8'),
                                               add_special_tokens=False)
            sentences =\
                [tokenizer.encode(s, add_special_tokens=False)
                 for s in sentence_tokenizer.tokenize(batch['abstract_ids'].numpy()[0].decode('utf-8'))]

            batch['output'] = label2index[batch['output'].numpy()[0].decode('utf-8')]
            batch['line_number'] = int(batch['line_number'].numpy()[0].decode('utf-8'))

            abstract_ids_list = []
            for i in range(len(sentences)):
                batch['abstract_ids'] = []
                for sentence in sentences[i:]:
                    if len(batch['title_ids'] + batch['abstract_ids'] + sentence) + 2 > MAX_LEN:
                        if len(batch['abstract_ids']) == 0:
                            batch['abstract_ids'] = sentence[:MAX_LEN - len(
                                batch['title_ids']) - 2]
                        break
                    else:
                        batch['abstract_ids'] += sentence
                abstract_ids_list.append(batch['abstract_ids'])

            if not SLIDING_WINDOW:
                batch['abstract_ids'] = \
                    abstract_ids_list[0]
                yield [batch]

            elif self.split == 'train':
                batch['abstract_ids'] = \
                    abstract_ids_list[np.random.random_integers(low=0, high=len(abstract_ids_list) - 1)]
                yield [batch]

            elif self.split == 'eval':
                for abstract_ids in abstract_ids_list:
                    batch['abstract_ids'] = abstract_ids
                    yield [batch]


    def __len__(self):
        return 1000000000

    def __iter__(self):
        return self.parse_file()

def create_tf_records(path_to_data, splits, k_fold):
    possible_empty_value = ['', 'Not available.', 'No abstract available.', 'NA', 'none',
                            'Not available', 'N/A.', 'No abstract available', 'N/A']
    writer = {s: tf.io.TFRecordWriter(DATA_PATH + 'dataset_' + s + '_fold=' + str(k_fold) + '.tfrecord') for s in SPLITS}
    count = {split: 0 for split in SPLITS}
    line_number2doc_id = {}
    for split in splits:
        with open(path_to_data + split + '_fold=' + str(k_fold) + '_v2.txt', encoding='utf-8') as f:
            line_number2doc_id[split] = {}
            for i, line in enumerate(f):
                id, label, title, journal, abstract = line.split('\t')
                line_number2doc_id[split][i] = id
                if title.strip() in possible_empty_value or abstract.strip() in possible_empty_value: continue
                count[split] += 1
                line_number = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(i).encode('utf-8')]))
                title_ids = tf.train.Feature(bytes_list=tf.train.BytesList(value=[(title + ' ' + journal).encode('utf-8')]))
                abstract_ids = tf.train.Feature(bytes_list=tf.train.BytesList(value=[abstract.encode('utf-8')]))
                label = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')]))
                features = {
                    'line_number': line_number,
                    'title_ids': title_ids,
                    'abstract_ids': abstract_ids,
                    'output': label
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer[split].write(example.SerializeToString())

    [writer[w].close() for w in writer.keys()]
    return count, line_number2doc_id

def extract_fn(example):
    features = {
        "line_number": tf.io.FixedLenFeature([], tf.string),
        "title_ids": tf.io.FixedLenFeature([], tf.string),
        "abstract_ids": tf.io.FixedLenFeature([], tf.string),
        "output": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    return example

def pytorch_convert(batch):

    dataset = [[], [], [], []]
    for b in batch:
        tmp = b[0]['title_ids'] + b[0]['abstract_ids']
        dataset[0].append([tokenizer.cls_token_id] +
                          tmp[:MAX_LEN-2] +
                          [tokenizer.sep_token_id])
        assert len(dataset[0][-1]) <= MAX_LEN
        dataset[1].append([1 for _ in dataset[0][-1]])
        dataset[2].append(b[0]['output'])
        dataset[3].append(b[0]['line_number'])

    input = torch.Tensor(tf.keras.preprocessing.sequence.pad_sequences(dataset[0], maxlen=MAX_LEN, dtype='long',
                                       value=0, truncating='post', padding='post'))
    mask = torch.Tensor(tf.keras.preprocessing.sequence.pad_sequences(dataset[1], maxlen=MAX_LEN, dtype='long',
                                      value=0, truncating='post', padding='post'))
    output = torch.Tensor(dataset[2])
    line_number = torch.Tensor(dataset[3])

    input, mask, output, line_number = input.type(torch.LongTensor),\
                          mask.type(torch.LongTensor),\
                          output.type(torch.LongTensor),\
                          line_number.type(torch.LongTensor)

    return [input, mask, output, line_number]

if __name__ == '__main__':

    main()