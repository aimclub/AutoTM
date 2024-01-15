from transformers import AutoTokenizer, AutoModel, utils, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
from autotm.preprocessing.text_preprocessing import process_dataset, PROCESSED_TEXT_COLUMN
import pandas as pd
import random

utils.logging.set_verbosity_error()


def preprocess_function(examples):
    return tokenizer(examples["processed_text"], truncation=True)


def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError(
                "The attention tensor does not have the correct number of dimensions. Make sure you set "
                "output_attentions=True when initializing your model."
            )
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        #             print(layer_attention[0])
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_attention(attention, head, layer):
    layer_attention = attention[layer]
    layer_attention = layer_attention.squeeze(0)
    head_attention = layer_attention[head]
    return head_attention


def num_layers(attention):
    return len(attention)


def num_heads(attention):
    return attention[0][0].size(0)


def format_special_chars(tokens):
    return [t.replace("Ġ", " ").replace("▁", " ").replace("</w>", "") for t in tokens]


def calculate_attention_head_confidence(
        attention,
        tokens,
        sentences,
        encoder_attention=None,
        decoder_attention=None,
        cross_attention=None,
        encoder_tokens=None,
        decoder_tokens=None,
        include_layers=None,
        sentence_b_start=None,
        layer=None,
        heads=None,
):
    attn_data = []
    if attention is not None:
        if tokens is None:
            raise ValueError("'tokens' is required")
        if (
                encoder_attention is not None
                or decoder_attention is not None
                or cross_attention is not None
                or encoder_tokens is not None
                or decoder_tokens is not None
        ):
            raise ValueError(
                "If you specify 'attention' you may not specify any encoder-decoder arguments. This"
                " argument is only for self-attention models."
            )
        if include_layers is None:
            include_layers = list(range(num_layers(attention)))
        attention = format_attention(attention, include_layers)
        if sentence_b_start is None:
            attn_data.append(
                {
                    "name": None,
                    "attn": attention.tolist(),
                    "left_text": tokens,
                    "right_text": tokens,
                }
            )
    if layer is not None and layer not in include_layers:
        raise ValueError(f"Layer {layer} is not in include_layers: {include_layers}")

    def get_attention_vals(attention, tokens, head_num, layer_num):
        attn_data = []
        n_heads = num_heads(attention)
        include_layers = list(range(num_layers(attention)))
        include_heads = list(range(n_heads))
        attention = get_attention(attention, head_num, layer_num)
        attention = attention[1:-1, 1:-1]
        tokens = tokens[1:-1]
        return attention, tokens


def get_model(model_name):
    if model_name == 'rubert-base-cased':
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased", output_attentions=True)
    if model_name == 'bert-base-cased-conversational':
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational")
        model = AutoModel.from_pretrained("DeepPavlov/bert-base-cased-conversational", output_attentions=True)
    return tokenizer, model


# Works long so better apply for small samples (< 1000 docs)
def build_graph(autotm_model, topic_labels,
                sampling=True, conn_strength=0.11):
    processed_df = pd.read_csv(os.path.join(autotm_model.working_dir_path, PROCESSED_TEXT_COLUMN))

    phi_df = autotm_model._model.get_phi()
    sentences = processed_df.processed_text.tolist()
    phi = phi_df[[i for i in list(phi_df) if i.startswith('main')]]

    data_dict = []

    if sampling:
        sentences = random.sample(sentences, 1000)

    for sentence in sentences:

        #     try:
        inputs = tokenizer.encode(sentence, return_tensors='pt')
        outputs = model(inputs)
        attention = outputs[-1]  # Retrieve attention from model outputs
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        res, tokens_new = get_attention_vals(attention, tokens, head_num=2, layer_num=0)
        print(res, tokens_new)

        try:
            v, i = torch.topk(res.flatten(), 5)
        except:
            continue
        idx = np.array(np.unravel_index(i.numpy(), res.shape)).T

        lemmatized_dict = {}
        all_phi_tokens = phi.index.tolist()
        # phi remove back
        for idx_items in idx:
            value = res[idx_items[0], idx_items[1]]
            # main topic strategy
            w1 = tokens_new[idx_items[0]]
            w2 = tokens_new[idx_items[1]]
            if w1 in lemmatized_dict:
                token1 = lemmatized_dict[w1]
            else:
                token1 = w1  # lemmatize_text(w1)
                lemmatized_dict[w1] = token1
            if w2 in lemmatized_dict:
                token2 = lemmatized_dict[w2]
            else:
                token2 = w2  # lemmatize_text(w2)
                lemmatized_dict[w2] = token2
            if token1 in all_phi_tokens and token2 in all_phi_tokens:
                if phi.loc[token1].sum() > 0 and phi.loc[token2].sum() > 0:
                    data_dict.append(
                        {'topic1': phi.loc[token1].idxmax(), 'topic2': phi.loc[token2].idxmax(), 'value': float(value)})

    connections_df = pd.DataFrame(data_dict)
    connections_df['topic1'] = connections_df['topic1'].apply(lambda x: topic_labels[x])
    connections_df['topic2'] = connections_df['topic2'].apply(lambda x: topic_labels[x])

    connections_df_filtered = connections_df[connections_df['value'] > conn_strength]
    connections_df_agg = connections_df_filtered.groupby(['topic1', 'topic2']).agg({'value': sum}).reset_index()
    connections_df_agg = connections_df_agg[connections_df_agg['topic1'] != connections_df_agg['topic2']]

    res_dict = {}
    for ix, row in connections_df_agg.iterrows():
        if (row['topic1'], row['topic2']) in res_dict:
            res_dict[(row['topic1'], row['topic2'])] += row['value']
        elif (row['topic2'], row['topic1']) in res_dict:
            res_dict[(row['topic2'], row['topic1'])] += row['value']
        else:
            res_dict[(row['topic1'], row['topic2'])] = row['value']

    nodes = set([i for res in list(res_dict.keys()) for i in res])

    return res_dict, nodes