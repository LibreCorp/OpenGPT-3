import torch


def test_transformer_forward():
    from opengpt3.modeling import Transformer

    # define small hyperparameters
    hparams = {
        'n_vocab': 100,
        'n_ctx': 16,
        'n_embd': 32,
        'n_layer': 2,
        'n_head': 4,
        'embd_pdrop': 0.0,
        'resid_pdrop': 0.0,
        'attn_pdrop': 0.0,
        'local_window': 16,
        'rate': 4,
        'layer_norm_epsilon': 1e-5,
    }
    model = Transformer(hparams)
    # batch size 2, sequence length 8
    input_ids = torch.randint(0, hparams['n_vocab'], (2, 8))
    # forward in training mode returns logits
    logits = model(input_ids)
    # logits shape: (batch*seq_len, n_vocab)
    assert logits.shape == (2 * 8, hparams['n_vocab'])


def test_transformer_forward_backward():
    from opengpt3.modeling import Transformer

    # define small hyperparameters
    hparams = {
        'n_vocab': 100,
        'n_ctx': 16,
        'n_embd': 32,
        'n_layer': 2,
        'n_head': 4,
        'embd_pdrop': 0.0,
        'resid_pdrop': 0.0,
        'attn_pdrop': 0.0,
        'local_window': 16,
        'rate': 4,
        'layer_norm_epsilon': 1e-5,
    }
    model = Transformer(hparams)
    # batch size 2, sequence length 8
    input_ids = torch.randint(0, hparams['n_vocab'], (2, 8))
    # forward pass to get logits
    logits = model(input_ids)
    # compute a dummy loss and backward through the network
    loss = logits.sum()
    loss.backward()
    # ensure gradients are computed for all trainable parameters
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)