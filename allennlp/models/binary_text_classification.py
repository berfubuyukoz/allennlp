from typing import Dict
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class TextClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 decoder: torch.nn.Module(),
                 vocab: Vocabulary
                 ) -> None:  # initializer: InitializerApplicator = InitializerApplicator(),
        # regularizer: Optional[RegularizerApplicator] = None
        # super().__init__(vocab, regularizer)
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.loss = torch.nn.CrossEntropyLoss()

        # initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None
                ) -> Dict[str, torch.Tensor]:
        embeddings = self.word_embeddings(text)
        mask = get_text_field_mask(text)
        state = self.encoder(embeddings, mask)  # called state since it returns final hidden state vector
        del embeddings
        logits = self.decoder(state)
        del state

        probabilities = F.softmax(logits, dim=-1)
        output_dict = {}
        output_dict["logits"] = logits
        output_dict["probabilities"] = probabilities
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            output_dict["loss"] = loss
        self.output_dict = output_dict
        self.labels = label

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        labels = self.labels
        labels_list = labels.squeeze(-1).cpu().data.numpy()
        predicted_labels = self.output_dict['predicted_labels']
        predicted_labels_as_int = [int(l) for l in predicted_labels]
        acc = accuracy_score(predicted_labels_as_int, labels_list)
        prf = precision_recall_fscore_support(predicted_labels_as_int, labels_list, average='macro')
        metrics = {}
        metrics['acc'] = acc
        metrics['prec'] = prf[0]
        metrics['rec'] = prf[1]
        metrics['fmacro'] = prf[2]
        return metrics

    def decode(self, reset: bool = False):
        predictions = self.output_dict['probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        self.output_dict['predicted_labels'] = labels
        metrics = self.get_metrics(reset=reset)
        self.output_dict['metrics'] = metrics