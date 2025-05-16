# text_analyzer.py
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"))
model.to(device)
model.eval()

def analyze_text(text):
    try:
        inputs = tokenizer(
            text, return_tensors='pt',
            max_length=512, truncation=True,
            padding='max_length'
        ).to(device)

        embeddings = model.distilbert.embeddings(inputs['input_ids'])
        embeddings.retain_grad()
        model.zero_grad()

        outputs = model.distilbert(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
        pooled = outputs.last_hidden_state[:, 0]
        pooled = model.pre_classifier(pooled)
        pooled = torch.relu(pooled)
        logits = model.classifier(pooled)

        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()[0]
        human_class_idx = 10
        ai_prob = 1 - probs[human_class_idx]

        model.zero_grad()
        logits[0].sum().backward()
        grads = embeddings.grad[0].cpu().numpy()
        importance = np.linalg.norm(grads, axis=-1)
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        highlighted = []
        current_word = ""
        current_score = 0

        for token, score, att in zip(tokens, importance, inputs['attention_mask'][0].cpu().numpy()):
            if token in ['[CLS]', '[SEP]', '[PAD]'] or att == 0:
                continue
            if token.startswith('##'):
                current_word += token[2:]
                current_score = max(current_score, score)
            else:
                if current_word:
                    color = f"rgba(255,0,0,{current_score})"
                    highlighted.append(f'<span style="background-color: {color}">{current_word}</span>')
                current_word = token
                current_score = score
        if current_word:
            color = f"rgba(255,0,0,{current_score})"
            highlighted.append(f'<span style="background-color: {color}">{current_word}</span>')

        return {
            'prediction': "AI-generated" if ai_prob > 0.5 else "Human",
            'probability': float(ai_prob if ai_prob > 0.5 else probs[human_class_idx]),
            'highlighted': ' '.join(highlighted)
        }

    except Exception as e:
        return {'error': str(e)}
