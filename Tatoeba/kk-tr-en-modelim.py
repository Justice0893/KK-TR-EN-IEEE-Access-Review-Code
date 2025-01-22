import opennmt

class MultiFeaturesTransformer(opennmt.models.Transformer):
    def __init__(self):
        super().__init__(
            source_inputter=opennmt.inputters.ParallelInputter(
                [
                    opennmt.inputters.WordEmbedder(embedding_size=512),
                    opennmt.inputters.WordEmbedder(embedding_size=20),
                ],
                reducer=opennmt.layers.ConcatReducer(),
            ),
            target_inputter=opennmt.inputters.WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1,
        )


model = MultiFeaturesTransformer