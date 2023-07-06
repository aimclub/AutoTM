import functools
import html
# Batches preparation
import os
import pickle
import re
import subprocess
from typing import Union

import artm
import click
import pandas as pd
import pymystem3
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

r_vk_ids = re.compile(r'(id{1}[0-9]*)')
r_num = re.compile(r'([0-9]+)')
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-@]')
r_white_space = re.compile(r'\s{2,}')
r_words = re.compile(r'\W+')
r_rus = re.compile(r'[а-яА-Я]\w+')
r_html = re.compile(r'(\<[^>]*\>)')
# clean texs from html
re1 = re.compile(r'  +')

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

stop = stopwords.words("russian") + [' '] + stopwords.words("english")

m = pymystem3.Mystem()
en_lemmatizer = WordNetLemmatizer()

def process_punkt(text):
    text = r_punct.sub(" ", text)
    text = r_vk_ids.sub(" ", text)
    text = r_num.sub(" ", text)
    text = r_white_space.sub(" ", text)
    return text.strip()


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def lemmatize_text(text, language='ru'):   # change language to en for english datasets
    try:
        text = new_html(text)
    except:
        return ''
    text = text.lower()
    text = process_punkt(text)
#     if language == 'ru':
#         text = re.findall(r_rus, text)
#         text = ' '.join(text)
    try:
        tokens = r_words.split(text)
    except:
        return ''
    tokens = (x for x in tokens if len(x) >= 3 and not x.isdigit())
    if language == 'ru':
        text = ' '.join(tokens)
        tokens = m.lemmatize(text)
    else:
        tokens = [get_lemma(token) for token in tokens]
    tokens = (x for x in tokens if len(x) > 3)
    tokens = (x for x in tokens if x not in stop)
    tokens = (x for x in tokens if x.isalpha())
    text = ' '.join(tokens)
    return text


def lemmatize_text_en(text):
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(text.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [en_lemmatizer.lemmatize(w) for w in text_rmstop]
    return ' '.join(text_stem)


def tokens_bigrams_to_text(tokens):
    return ' '.join(['_'.join(tok.split()) for tok in tokens])


def text_to_tokens(text):
    return text.split()


def get_tokens_count(text):
    return len(text.split())


def remove_more_html(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('<', ' ').replace('>', ' ').replace('#36;', '$').replace(
        '\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('img', ' ').replace('class', ' ').replace(
        'src', ' ').replace('alt', ' ').replace('email', ' ').replace('icq', ' ').replace(
        'href', ' ').replace('mem', ' ').replace('link', ' ').replace('mention', ' ').replace(
        'onclick', ' ').replace('icq', ' ').replace('onmouseover', ' ').replace('post', ' ').replace(
        'local', ' ').replace('key', ' ').replace('target', ' ').replace('amp', ' ').replace(
        'section', ' ').replace('search', ' ').replace('css', ' ').replace('style', ' ').replace(
        'cc', ' ').replace("img", ' ').replace("expand", ' ').replace('\n', ' ').replace(
        'dnum', ' ').replace('xnum', ' ').replace('nnum', ' ')
    return re1.sub(' ', html.unescape(x))


def clear_url(text):
    return re.sub(url_pattern, ' ', text)


def new_html(text):
    text = r_html.sub("", text)
    return text


def tokens_num(text):
    return len(text.split(' '))


def get_words_dict(text, stop_list):
    all_words = text
    words = sorted(set(all_words) - stop_list)

    return {w: all_words.count(w) for w in words}


def return_string_part(name_type, text):
    tokens = text.split()
    tokens = [item for item in tokens if item != '']
    tokens_dict = get_words_dict(tokens, set())

    return " |" + name_type + ' ' + ' '.join(['{}:{}'.format(k, v) for k, v in tokens_dict.items()])


def prepare_voc(batches_dir, vw_path, data_path, column_name='processed_text'):
    print('Starting...')
    with open(vw_path, 'w', encoding='utf8') as ofile:
        num_parts = 0
        try:
            for file in os.listdir(data_path):
                if file.startswith('part'):
                    print('part_{}'.format(num_parts), end='\r')
                    if file.split('.')[-1] == 'csv':
                        part = pd.read_csv(os.path.join(data_path, file))
                    else:
                        part = pd.read_parquet(os.path.join(data_path, file))
                    part_processed = part[column_name].tolist()
                    for text in part_processed:
                        result = return_string_part('@default_class', text)
                        ofile.write(result + '\n')
                    num_parts += 1

        except NotADirectoryError:
            print('part 1/1')
            part = pd.read_csv(data_path)
            part_processed = part[column_name].tolist()
            for text in part_processed:
                result = return_string_part('@default_class', text)
                ofile.write(result + '\n')

    print(' batches {} \n vocabulary {} \n are ready'.format(batches_dir, vw_path))


def prepare_batch_vectorizer(batches_dir, vw_path, data_path, column_name='processed_text'):
    #     if not glob.glob(os.path.join(batches_dir, "*")):
    prepare_voc(batches_dir, vw_path, data_path, column_name=column_name)
    batch_vectorizer = artm.BatchVectorizer(data_path=vw_path,
                                            data_format="vowpal_wabbit",
                                            target_folder=batches_dir,
                                            batch_size=100)
    #     else:
    #         batch_vectorizer = artm.BatchVectorizer(data_path=batches_dir, data_format='batches')

    return batch_vectorizer


def vocab_preparation(VOCAB_PATH, DICTIONARY_PATH):
    if not os.path.exists(VOCAB_PATH):
        with open(DICTIONARY_PATH, 'r') as dictionary_file:
            with open(VOCAB_PATH, 'w') as vocab_file:
                dictionary_file.readline()
                dictionary_file.readline()
                for line in dictionary_file:
                    elems = re.split(', ', line)
                    vocab_file.write(' '.join(elems[:2]) + '\n')


def preparing_cooc_dict(BATCHES_DIR, WV_PATH, VOCAB_PATH, COOC_DICTIONARY_PATH, cooc_file_path_tf, cooc_file_path_df,
                         ppmi_dict_tf, ppmi_dict_df):
    cmd_args = [
        "bigartm",
        "-c", WV_PATH, "-v", VOCAB_PATH, "--cooc-window", "10", "--write-cooc-tf", cooc_file_path_tf,
        "--write-cooc-df", cooc_file_path_df, "--write-ppmi-tf", ppmi_dict_tf, "--write-ppmi-df", ppmi_dict_df

    ]

    cproc = subprocess.run(cmd_args, capture_output=True)

    assert cproc.returncode == 0
    cooc_dict = artm.Dictionary()
    cooc_dict.gather(
        data_path=BATCHES_DIR,
        cooc_file_path=ppmi_dict_tf,
        vocab_file_path=VOCAB_PATH,
        symmetric_cooc_values=True)
    cooc_dict.save_text(COOC_DICTIONARY_PATH)


def prepare_artm_files(BATCHES_DIR, WV_PATH, DOCUMENTS_TO_BATCH_PATH, DICTIONARY_PATH,
                       VOCAB_PATH, COOC_DICTIONARY_PATH, cooc_file_path_tf,
                       cooc_file_path_df, ppmi_dict_tf, ppmi_dict_df, MUTUAL_INFO_DICT_PATH):
    batch_vectorizer = prepare_batch_vectorizer(BATCHES_DIR, WV_PATH, DOCUMENTS_TO_BATCH_PATH)

    my_dictionary = artm.Dictionary()
    my_dictionary.gather(data_path=BATCHES_DIR, vocab_file_path=WV_PATH)
    my_dictionary.filter(min_df=3, class_id='text')
    my_dictionary.save_text(DICTIONARY_PATH)

    vocab_preparation(VOCAB_PATH, DICTIONARY_PATH)
    preparing_cooc_dict(BATCHES_DIR, WV_PATH, VOCAB_PATH,
                         COOC_DICTIONARY_PATH, cooc_file_path_tf,
                         cooc_file_path_df, ppmi_dict_tf,
                         ppmi_dict_df)

    mutual_info_dict = mutual_info_dict_preparation(ppmi_dict_tf)
    with open(MUTUAL_INFO_DICT_PATH, 'wb') as handle:
        pickle.dump(mutual_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def mutual_info_dict_preparation(fname):

    tokens_dict = {}

    with open(fname) as handle:
        for ix, line in tqdm(enumerate(handle)):
            list_of_words = line.strip().split()
            word_1 = list_of_words[0]
            for word_val in list_of_words[1:]:
                word_2, value = word_val.split(':')
                tokens_dict['{}_{}'.format(word_1, word_2)] = float(value)
                tokens_dict['{}_{}'.format(word_2, word_1)] = float(value)
    return tokens_dict


def dataset_preprocessing(dataset: Union[str, DataFrame], col_to_process, save_path, language='ru'):
    data = pd.read_csv(dataset) if isinstance(dataset, str) else dataset
    lemm_txt = functools.partial(lemmatize_text, language=language)
    data['processed_text'] = data[col_to_process].apply(lemm_txt)
    data['tokens_len'] = data['processed_text'].apply(tokens_num)
    data = data[data['tokens_len'] > 3]
    data.to_csv(save_path, index=None)
    print('Saved to {}'.format(save_path))


def do_preprocessing(dataset: Union[str, DataFrame], dataset_path: str, language: str = 'ru'):
    BATCHES_DIR = os.path.join(dataset_path, 'batches')
    WV_PATH = os.path.join(dataset_path, 'test_set_data_voc.txt')
    COOC_DICTIONARY_PATH = os.path.join(dataset_path, 'cooc_dictionary.txt')
    DICTIONARY_PATH = os.path.join(dataset_path, 'dictionary.txt')
    VOCAB_PATH = os.path.join(dataset_path, 'vocab.txt')
    cooc_file_path_df = os.path.join(dataset_path, 'cooc_df.txt')
    cooc_file_path_tf = os.path.join(dataset_path, 'cooc_tf.txt')
    ppmi_dict_df = os.path.join(dataset_path, 'ppmi_df.txt')
    ppmi_dict_tf = os.path.join(dataset_path, 'ppmi_tf.txt')
    MUTUAL_INFO_DICT_PATH = os.path.join(dataset_path, 'mutual_info_dict.pkl')
    col_to_process = 'text'
    save_path = os.path.join(dataset_path, 'dataset_processed.csv')

    dataset_preprocessing(dataset, col_to_process, save_path, language)

    prepare_artm_files(BATCHES_DIR, WV_PATH, save_path, DICTIONARY_PATH,
                       VOCAB_PATH, COOC_DICTIONARY_PATH, cooc_file_path_tf,
                       cooc_file_path_df, ppmi_dict_tf, ppmi_dict_df, MUTUAL_INFO_DICT_PATH)


@click.group()
def cli():
    pass


@cli.command(help="processes an arbitrary dataset that is presented as a CSV file with 'text' column")
@click.option("--data", "dataset_path", required=True, type=str,
              help='Path to a CSV file of the dataset to be processed')
@click.option("--out", "out_path", required=True, type=str, help='Output folder to be filled with all necessary files')
@click.option("--lang", default='en', type=str, help='Language of the texts')
def make_dataset(dataset_path: str, out_path: str, lang: str):
    do_preprocessing(dataset_path, out_path, lang)


@cli.command(help="downloads 20newsgroup dataset from the web, "
                  "cuts the first 100 records and processes it as a regulat 'make-dataset' command")
@click.option("--out", "out_path", default="/tmp/tiny_dataset", show_default=True, type=str,
              help='Output folder to be filled with all necessary files')
@click.option("--lang", default='en', type=str, help='Language of the texts')
def make_test_dataset(out_path: str, lang: str):
    os.makedirs(out_path, exist_ok=True)

    bunch = fetch_20newsgroups(data_home=out_path)
    dataset = pd.DataFrame(bunch['data'][:100], columns=['text'])

    do_preprocessing(dataset, out_path)


if __name__ == "__main__":
    cli()
