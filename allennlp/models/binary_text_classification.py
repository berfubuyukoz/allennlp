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

    def _confusion(self, truth, predictions):
        confusion_matrix = torch.zeros(self.args.nclasses, self.args.nclasses)
        for t, p in zip(truth.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        tp = confusion_matrix[1,1].item()
        fp = confusion_matrix[0,1].item()
        tn = confusion_matrix[0,0].item()
        fn = confusion_matrix[1,0].item()
        return tp,fp,tn,fn

    def _get_scores(self, truth, predictions):
        tp,fp,tn,fn = self._confusion(truth, predictions)
        prot_recall = tp/(tp+fn) if (tp+fn) else 0
        prot_precision = tp/(tp+fp) if (tp+fp) else 0
        prot_f1_score = 2 * prot_precision * prot_recall / (prot_precision + prot_recall) if (prot_precision + prot_recall) else 0

        non_prot_recall = tn/(tn+fp) if (tn+fp) else 0
        non_prot_precision = tn/(tn+fn) if (tn+fn) else 0
        non_prot_f1_score = 2 * non_prot_precision * non_prot_recall / (non_prot_precision + non_prot_recall) if (non_prot_precision + non_prot_recall) else 0

        # Take average of two class scores (macro)
        f1_macro = (prot_f1_score + non_prot_f1_score)/2
        prec_macro = (prot_precision + non_prot_precision)/2
        recall_macro = (prot_recall + non_prot_recall)/2
        acc = (tp + tn)/(tp+fp+tn+fn)
        return acc,prec_macro,recall_macro,f1_macro, prot_precision, prot_recall, prot_f1_score

    def get_metrics(self) -> Dict[str, float]:
        labels = self.labels
        # labels_list = labels.squeeze(-1).cpu().data.numpy()
        # predicted_labels = self.output_dict['predicted_labels'].squeeze(-1).cpu().data.numpy()
        # predicted_labels_as_int = [int(l) for l in predicted_labels]
        # acc = accuracy_score(labels_list, predicted_labels)
        # prf = precision_recall_fscore_support(labels_list, predicted_labels, average='macro')
        acc,prec,rec,fmacro, pos_prec, pos_rec, pos_fmacro = self._get_scores(labels, self.output_dict['predicted_labels'])
        metrics = {}
        metrics['acc'] = acc
        metrics['prec'] = prec
        metrics['rec'] = rec
        metrics['fmacro'] = fmacro
        metrics['pos_prec'] = pos_prec
        metrics['pos_rec'] = pos_rec
        metrics['pos_fmacro'] = pos_fmacro
        return metrics

    def decode(self):
        probabilities = self.output_dict['probabilities']
        predictions = torch.argmax(probabilities, dim=1)
        # predicted_labels = [self.vocab.get_token_from_index(x, namespace="labels")
        #           for x in predictions]
        self.output_dict['predicted_labels'] = predictions
        metrics = self.get_metrics()
        self.output_dict['metrics'] = metrics