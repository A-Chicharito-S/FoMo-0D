import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from pfns.transformer import TransformerModel
from pfns import encoders

class FoMo0DHub(nn.Module, PyTorchModelHubMixin):
    """
    Minimal inference-only wrapper that builds FoMo-0D from config params and
    exposes it to the Hub via PyTorchModelHubMixin.
    """
    def __init__(
        self,
        # --- core arch / hyperparams to intialize the ckpt for FoMo-0D ---
        num_features: int = 100,
        emsize: int = 256,
        nhid: int = 512,
        nlayers: int = 4,
        nhead: int = 4,
        num_R: int = 500,
        dropout: float = 0.0,
        seq_len: int = 5000,
        efficient_eval_masking: bool = True,
        num_global_att_tokens: int = 0,
        num_class: int = 2,
        # decoders and optional submodules
        decoder_dict: dict | None = {},
        initializer = None,
        # anything else FoMo-0D accepts can be forwarded via **model_extra_args
        **model_extra_args,
    ):
        super().__init__()

        decoder_dict = decoder_dict if decoder_dict else {'standard': (None, num_class)}

        # mirror your make_model_od defaults

        self.model = TransformerModel(
            encoder=encoders.get_normalized_uniform_encoder(encoders.Linear)(num_features, emsize),
            nhead=nhead,
            ninp=emsize,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            style_encoder=None,
            y_encoder=encoders.Linear(1, emsize=emsize),
            input_normalization=False,
            pos_encoder=None,
            decoder_dict={'standard': (None, num_class)},
            init_method=initializer,
            efficient_eval_masking=efficient_eval_masking,
            decoder_once_dict={},
            num_global_att_tokens=num_global_att_tokens,
            model_para_dict={'num_R': num_R, 'last_layer_no_R': True},
            **model_extra_args
        )

        self.model.criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none', ignore_index=-33)

        # Saved to config.json on the Hub by the mixin:
        self.config = dict(
            num_features=num_features,
            emsize=emsize, 
            nhid=nhid, 
            nlayers=nlayers, 
            nhead=nhead, 
            dropout=dropout, 
            seq_len=seq_len,
            num_R=num_R
    )

    def forward(self, train_x, test_x):
        out = self.model(train_x, None, test_x) # (num_test_x, bs, num_classes=2)
        return out
