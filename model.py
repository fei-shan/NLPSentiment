from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
# from datasets import load_dataset


class Model:
    def __init__(self, model_checkpoint="distilbert-base-uncased"):
        self.model_checkpoint = model_checkpoint
        self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        num_parameters = self.model.num_parameters() / 1_000_000
        print(f"'>>> number of parameters: {round(num_parameters)}M'")
        print(f"'>>> BERT number of parameters: 110M'")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_logits = self.model(**inputs).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        # Pick the [MASK] candidates with the highest logits
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        for token in top_5_tokens:
            print(f"'>>> {text.replace(self.tokenizer.mask_token, self.tokenizer.decode([token]))}'")


if __name__ == '__main__':
    model = Model()
    model.predict('This is a great [MASK].')
