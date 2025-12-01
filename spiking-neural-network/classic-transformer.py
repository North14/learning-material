import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torchviz import make_dot


class ClassicTransformer(nn.Module):
    def __init__(self, input_size, model_dim, num_heads, num_layers, output_size, dropout=0.1):
        super(ClassicTransformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_size, model_dim)
        encoder_layers = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.model_dim)
        src = self.dropout(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output
def create_transformer(input_size, model_dim, num_heads, num_layers, output_size, dropout=0.1):
    return ClassicTransformer(input_size, model_dim, num_heads, num_layers, output_size, dropout)
if __name__ == "__main__":
    input_size = 100
    model_dim = 64
    num_heads = 8
    num_layers = 3
    output_size = 10
    model = create_transformer(input_size, model_dim, num_heads, num_layers, output_size)
    sample_input = torch.randn(10, 32, input_size)  # (sequence_length, batch_size, input_size)
    print(sample_input)
    output = model(sample_input)
    print(output)
    # Graph visualization and analysis
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('transformer_graph')