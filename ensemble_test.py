'''
Code for computing ensembles performances

'''
import glob, numpy as np, logging, operator, sys
from sklearn.metrics import classification_report
from scipy.special import softmax
from collections import defaultdict

path_to_models = sys.argv[1]
evaluation_level = sys.argv[2]

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
label2index = defaultdict(list)
for key, value in index2label.items():
    label2index[value].append(key)

index2binary = {
    **{i: 'ORIGINAL'  for i in range(len(label2index)) if 'EPI' in index2label[i] or 'BASIC' in index2label[i]},
    **{i: 'NON-ORIGINAL'  for i in range(len(label2index)) if not ('EPI' in index2label[i] or 'BASIC' in index2label[i])}
}
binary2index = defaultdict(list)
for key, value in index2binary.items():
    binary2index[value].append(key)


index2triple = {
    **{i: 'EPI'  for i in range(len(label2index)) if 'EPI' in index2label[i]},
    **{i: 'BASIC'  for i in range(len(label2index)) if 'BASIC' in index2label[i]},
    **{i: 'OTHER'  for i in range(len(label2index)) if 'EPI' not in index2label[i] and 'BASIC' not in index2label[i]}
}
triple2index = defaultdict(list)
for key, value in index2triple.items():
    triple2index[value].append(key)


model_paths = [f for f in glob.glob(path_to_models + "*") if '.txt' not in f and '.png' not in f]
strategies = ['voting', 'prob_sum']

predictions = {'Gold': {}}
predictions_evaluation_level = {}
latex_dict = {}
for MODEL_DIR in model_paths:
    model_name = MODEL_DIR.split('/')[-1]
    predictions[model_name] = {}
    latex_dict[model_name] = {}
    fold_predictions_files = list(glob.glob(MODEL_DIR + "/fold=*"))
    for predictions_path in fold_predictions_files:
        with open(predictions_path) as f:
            predictions_file = f.read().strip()

        for line in predictions_file.split('\n'):
            id, label, *logits = line.split('\t')
            y_pred = np.array([float(l) for l in logits])
            if id in predictions[model_name] :
                predictions[model_name][id]['prob'] += softmax(y_pred)
            else:
                predictions[model_name][id] = {'prob': softmax(y_pred)}

            if id not in predictions['Gold']:
                predictions['Gold'][id] = label2index[label][0]

    predictions_evaluation_level[model_name] = {}
    predictions_evaluation_level['Gold'] = {}
    y_true = []
    y_pred = []
    if evaluation_level == 'subclass':
        dictionary_to_use = index2label
        inverted_to_use = label2index
    elif evaluation_level == 'class':
        dictionary_to_use = index2triple
        inverted_to_use = triple2index
    elif evaluation_level == 'binary':
        dictionary_to_use = index2binary
        inverted_to_use = binary2index
        true_label2index = {'ORIGINAL': 1, 'NON-ORIGINAL': 0}
        true_index2label = {v: k for k, v in true_label2index.items()}


    for id in predictions[model_name].keys():
        y_true.append(true_index2label[predictions['Gold'][id]])
        pred = {label: 0 for label in inverted_to_use.keys()}
        for label in inverted_to_use.keys():
            for index in inverted_to_use[label]:
                pred[label] += predictions[model_name][id]['prob'][index]
        y_pred.append(max(pred.items(), key=operator.itemgetter(1))[0])
        predictions_evaluation_level[model_name][id] = y_pred[-1]
        predictions_evaluation_level['Gold'][id] = y_true[-1]

    report = classification_report(y_true, y_pred, digits=4)
    print(model_name)
    print(report)

    latex_dict[model_name][evaluation_level] = classification_report(y_true, y_pred, output_dict=True)
    for id in predictions[model_name].keys():
        predictions[model_name][id]['prob'] /= len(fold_predictions_files)


def ensemble(ensemble, models_names, evaluation_level, strategy, voting_threshold=None):

    if evaluation_level == 'subclass':
        dictionary_to_use = index2label
        inverted_to_use = label2index
        default_pred = 'Comment, editorial, ..., non-original'
    elif evaluation_level == 'class':
        dictionary_to_use = index2triple
        inverted_to_use = triple2index
        default_pred = 'OTHER'
    elif evaluation_level == 'binary':
        dictionary_to_use = index2binary
        inverted_to_use = binary2index
        default_pred = 'NON-ORIGINAL'
        true_label2index = {'ORIGINAL': 1, 'NON-ORIGINAL': 0}
        true_index2label = {v: k for k, v in true_label2index.items()}

    if strategy == 'voting': majority = int(len(models_names)/2) + 1

    predictions_evaluation_level = {}
    y_pred_dict, y_true_dict = {}, {}
    for model_number in list(ensemble):
        for id in predictions[models_names[model_number]].keys():

            if strategy == 'prob_sum':
                if id not in y_pred_dict:
                    y_pred_dict[id] = np.array(predictions[models_names[model_number]][id]['prob'])
                    y_true_dict[id] = predictions['Gold'][id]

                else:
                    y_pred_dict[id] += predictions[models_names[model_number]][id]['prob']

            elif strategy == 'voting':
                pred = {label: 0 for label in inverted_to_use.keys()}
                for label in inverted_to_use.keys():
                    for index in inverted_to_use[label]:
                        pred[label] += predictions[models_names[model_number]][id]['prob'][index]

                pred = {l: pred[l] for l in pred.keys()}
                pred = [l for l in pred.keys() if pred[l] >= voting_threshold]

                if id not in y_pred_dict:
                    y_pred_dict[id] = {}
                    y_true_dict[id] = predictions['Gold'][id]

                if len(pred) == 0:
                    continue
                else:
                    pred = pred[0]

                if pred not in y_pred_dict[id]:
                    y_pred_dict[id][pred] = 1
                else:
                    y_pred_dict[id][pred] += 1


    if strategy == 'prob_sum':
        for id in y_pred_dict.keys():
            y_pred_dict[id] /= 5

        y_pred = []
        y_true = []
        for id in y_pred_dict.keys():
            pred = {label: 0 for label in inverted_to_use.keys()}
            for label in inverted_to_use.keys():
                for index in inverted_to_use[label]:
                    pred[label] += y_pred_dict[id][index]
            pred = [l for l in pred.keys() if pred[l] >= voting_threshold]
            if len(pred) == 0:
                continue
            else:
                pred = pred[0]

            y_pred.append(pred)
            y_true.append(true_index2label[y_true_dict[id]])
            predictions_evaluation_level[id] = y_pred[-1]

        return [y_true, y_pred, predictions_evaluation_level]


    elif strategy == 'voting':
        voting = {}
        y_pred = []
        y_true = []
        no_pred_count = 0
        for id in y_pred_dict.keys():
            voting[id] = {}
            pred = None
            for label in y_pred_dict[id].keys():
                voting[id][label] = y_pred_dict[id][label]
                if y_pred_dict[id][label] >= majority:
                    pred = label
                    break
            if not pred:
                no_pred_count += 1
                pred = default_pred
            y_pred.append(pred)
            y_true.append(true_index2label[y_true_dict[id]])
            predictions_evaluation_level[id] = y_pred[-1]

        return [y_true, y_pred, predictions_evaluation_level, voting]



relative_improvement = [0]
metrics = {
    'f1-score': [0],
    'precision': [0],
    'recall': [0]
}
positive_recall_per_doc_taken = [0]
voting_thresholds = []
thresholds_n_votes = []
n_documents = []

results_to_plot = {}
with open(path_to_models + 'latex_results.txt', 'w') as f:
    for voting_threshold in [i/1000 for i in range(500, 1000, 50)] + [.99]:
        for strategy in strategies:
            models_names = [model_name for model_name in predictions.keys() if model_name != 'Gold']
            if strategy == 'voting':
                y_true, y_pred, tmp, voting = ensemble(tuple(i for i in range(len(models_names))), models_names,
                                                       evaluation_level=evaluation_level, strategy='voting',
                                                       voting_threshold=voting_threshold)
            else:
                y_true, y_pred, tmp = ensemble(tuple(i for i in range(len(models_names))), models_names,
                                               evaluation_level=evaluation_level, strategy=strategy,
                                                       voting_threshold=voting_threshold)
            predictions_evaluation_level[strategy] = tmp
            report = classification_report(y_true, y_pred, digits=4)

            latex_dict[strategy] = classification_report(y_true, y_pred, output_dict=True)
            if strategy == 'prob_sum':
                f.write('---\n')
                f.write('prob_sum;' + str(voting_threshold) + ' \n')
                for k in latex_dict[strategy].keys():
                    if k == 'accuracy': continue
                    f.write(k + ' & ' + ' & '.join(['{:.2f}'.format(latex_dict[strategy][k][m] * 100)
                                                    if latex_dict[strategy][k][m]<1 else str(latex_dict[strategy][k][m])
                                                    for m in ['precision', 'recall', 'f1-score', 'support']]) + '\\\\ \hline \n')
            print(strategy + ': ' + evaluation_level)
            print(models_names)
            print(report)

        for with_probsum_recovery in [False]:
            for threshold in [3, 4, 5, 'majority', 'unanimity']:

                n_recovered = 0
                n_voted = 0
                y_pred = []
                y_true = []
                only_original = []
                for id in voting.keys():
                    has_no_pred = True
                    if voting[id]:
                        if threshold == 'majority':
                            t = np.max([int(sum([voting[id][v] for v in voting[id].keys()])/2) + 1, 2])
                        elif threshold == 'unanimity':
                            t = np.max([sum([voting[id][v] for v in voting[id].keys()]), 2])
                        else:
                            t = threshold

                    else:
                        t = 0

                    for label in voting[id].keys():
                        if voting[id][label] >= t:
                            n_voted += 1
                            y_pred.append(label)
                            has_no_pred = False
                            only_original.append('ORIGINAL')
                            y_true.append(predictions_evaluation_level['Gold'][id])
                            break

                    # if no prediction -> take prob_sum
                    if has_no_pred and with_probsum_recovery:
                        n_recovered += 1
                        y_pred.append(predictions_evaluation_level['prob_sum'][id])
                        only_original.append('ORIGINAL')
                        y_true.append(predictions_evaluation_level['Gold'][id])

                f1_only_original = classification_report(y_true, only_original, output_dict=True)['weighted avg']['f1-score']
                tmp = classification_report(y_true, y_pred, output_dict=True)
                f.write('---\n')
                f.write('voting;' + str(voting_threshold) + ' - ' + str(threshold) + ' \n')
                for k in tmp.keys():
                    if k == 'accuracy': continue
                    f.write(k + ' & ' + ' & '.join(['{:.2f}'.format(tmp[k][m] * 100)
                                                    if tmp[k][m] < 1 else str(tmp[k][m])
                                                    for m in
                                                    ['precision', 'recall', 'f1-score', 'support']]) + '\\\\ \hline \n')
                metrics['f1-score'].append(tmp['weighted avg']['f1-score'])
                metrics['recall'].append(tmp['ORIGINAL']['recall'])
                metrics['precision'].append(tmp['ORIGINAL']['precision'])
                positive_recall_per_doc_taken.append(tmp['ORIGINAL']['recall'])
                n_documents.append(tmp['weighted avg']['support'])
                thresholds_n_votes.append(threshold)
                voting_thresholds.append(voting_threshold)
                relative_improvement.append((metrics['f1-score'][-1] / f1_only_original) - 1)

                proportion = 'relative improvement of: ' + '{:.2f}'.format(relative_improvement[-1] * 100) + '; voting_threshold:' +\
                             str(voting_threshold) + '; votes:' + str(int(n_voted)) + '/recovered:' + str(int(n_recovered)) + '\n'

                report = classification_report(y_true, y_pred, digits=4)
                report = classification_report(y_true, only_original, digits=4)

from tqdm import tqdm
f1_scores = {m: {'prob_sum': [], 'voting_majority': [], 'voting_unanimity': []} for m in ['micro', 'macro']}
subset_prop = {'prob_sum': [], 'voting_majority': [], 'voting_unanimity': []}
for voting_threshold in tqdm(range(500, 1000)):
    voting_threshold /= 1000
    for strategy in strategies:
        models_names = [model_name for model_name in predictions.keys() if model_name != 'Gold']
        if strategy == 'voting':
            y_true, y_pred, tmp, voting = ensemble(tuple(i for i in range(len(models_names))), models_names,
                                                   evaluation_level=evaluation_level, strategy='voting',
                                                   voting_threshold=voting_threshold)
        else:
            y_true, y_pred, tmp = ensemble(tuple(i for i in range(len(models_names))), models_names,
                                           evaluation_level=evaluation_level, strategy=strategy,
                                                   voting_threshold=voting_threshold)
        predictions_evaluation_level[strategy] = tmp
        report = classification_report(y_true, y_pred, digits=4)

        latex_dict[strategy] = classification_report(y_true, y_pred, output_dict=True)
        if strategy == 'prob_sum':
            f1_scores['micro']['prob_sum'].append(latex_dict[strategy]['weighted avg']['f1-score'])
            f1_scores['macro']['prob_sum'].append(latex_dict[strategy]['macro avg']['f1-score'])
            subset_prop['prob_sum'].append(latex_dict[strategy]['weighted avg']['support'] / 968)

    for with_probsum_recovery in [False]:
        for threshold in [3, 5]:

            n_recovered = 0
            n_voted = 0
            y_pred = []
            y_true = []
            only_original = []
            for id in voting.keys():
                has_no_pred = True
                if voting[id]:
                    if threshold == 'majority':
                        t = np.max([int(sum([voting[id][v] for v in voting[id].keys()])/2) + 1, 2])
                    elif threshold == 'unanimity':
                        t = np.max([sum([voting[id][v] for v in voting[id].keys()]), 2])
                    else:
                        t = threshold

                else:
                    t = 0

                for label in voting[id].keys():
                    if voting[id][label] >= t:
                        n_voted += 1
                        y_pred.append(label)
                        has_no_pred = False
                        only_original.append('ORIGINAL')
                        y_true.append(predictions_evaluation_level['Gold'][id])
                        break

            try:
                f1_only_original = classification_report(y_true, only_original, output_dict=True)['weighted avg']['f1-score']
                tmp = classification_report(y_true, y_pred, output_dict=True)
                if threshold == 3:
                    f1_scores['micro']['voting_majority'].append(tmp['weighted avg']['f1-score'])
                    f1_scores['macro']['voting_majority'].append(tmp['macro avg']['f1-score'])
                    subset_prop['voting_majority'].append(tmp['weighted avg']['support'] / 968)

                elif threshold == 5:
                    f1_scores['micro']['voting_unanimity'].append(tmp['weighted avg']['f1-score'])
                    f1_scores['macro']['voting_unanimity'].append(tmp['macro avg']['f1-score'])
                    subset_prop['voting_unanimity'].append(tmp['weighted avg']['support'] / 968)

            except:
                print('error')