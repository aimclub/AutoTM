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
