from typing import Dict
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
import numpy as np

class TextClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 ) -> None:  # initializer: InitializerApplicator = InitializerApplicator(),
        # regularizer: Optional[RegularizerApplicator] = None
        # super().__init__(vocab, regularizer)
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.classifier_input_dim = self.encoder.get_output_dim()
        self.classification_layer = torch.nn.Linear(self.classifier_input_dim, self.num_classes)
        self.accuracy = CategoricalAccuracy()
        self.fmacro = FBetaMeasure(average='macro')
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {}
        self.metrics["accuracy"] = self.accuracy
        self.metrics["fmacro"] = self.fmacro
        # initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None
                ) -> Dict[str, torch.Tensor]:
        embeddings = self.word_embeddings(text)
        mask = get_text_field_mask(text)
        state = self.encoder(embeddings, mask)  # called state since it returns final hidden state vector
        del embeddings
        logits = self.classification_layer(state)
        del state

        probabilities = F.softmax(logits)
        output_dict = {}
        output_dict["logits"] = logits
        output_dict["probabilities"] = probabilities
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            output_dict["loss"] = loss
            self.accuracy(logits, label.squeeze(-1))
            self.fmacro(logits, label.squeeze(-1))
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # compute accuracy using intermediate vars that were already updated behind the scene by CategoricalAccuracy.__call__().
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        metrics = self.get_metrics()
        output_dict['metrics'] = metrics
        return output_dict