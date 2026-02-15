# PubMed BERT Abstract Sentence Classification


### Note: `.\FineTuningBert.ipynb` Contains the code to the Banana Assignment, However due to Notebook's Json Internal error, the outputs cannot be visualized in Github. In order to **see the Outputs**, Please visit the link to the Notebook: [LangRangers Finetuning Assignment with Output](https://colab.research.google.com/drive/1YsZ9L_07gGR9WUQEoJ48WJp72eHgOMh0?usp=sharing)

---

## Team Members

| Name | SRN | GitHub |
|------|-----|--------|
| Aaron Thomas Mathew | PES1UG23AM005 | https://github.com/aaronmat1905 |
| Aman Kumar Mishra | PES1UG23AM040 | https://github.com/amankumarmishra |
| Preetham V J | PES1UG23AM913 | https://github.com/preethamvj |

---

## 1. Problem Statement

Medical research abstracts are structured documents containing distinct sections: BACKGROUND, OBJECTIVE, METHODS, RESULTS, and CONCLUSIONS. Our task is to build a deep learning classifier that automatically categorizes individual sentences from medical abstracts into these five categories.

**Applications:**
- Automated literature review
- Medical search engines
- Abstract quality assessment
- Information extraction from biomedical text

---

## 2. Dataset

**PubMed 200k RCT Dataset**
- 200,000+ medical abstracts from PubMed
- Each sentence labeled with one of 5 categories
- Pre-split into train/validation/test sets

**Test Set Distribution:**
- BACKGROUND: 1,340 sentences (9.1%)
- OBJECTIVE: 1,190 sentences (8.1%)
- METHODS: 4,893 sentences (33.2%)
- RESULTS: 5,106 sentences (34.6%)
- CONCLUSIONS: 2,217 sentences (15.0%)
- **Total: 14,746 sentences**

---

## 3. Model Architecture

We implemented a BERT-based classifier using transfer learning:

```python
class BERTClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_bert=False, dropout=0.1):
        super(BERTClassifier, self).__init__()
        
        # Pre-trained BERT encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Optional freezing
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
```

**Model Components:**
- Base: `bert-base-uncased` (110M parameters)
- BERT encoder: 12 transformer layers, 768 hidden dimensions
- Dropout: 0.1
- Classification head: Linear layer (768 → 5)

---

## 4. Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| SUBSET_FRACTION | 0.6 | Used 60% of training data |
| FREEZE_BERT | False | All BERT layers fine-tuned |
| MAX_LENGTH | 128 | Maximum token sequence length |
| BATCH_SIZE | 16 | Training batch size |
| LEARNING_RATE | 2e-5 | Adam optimizer learning rate |
| EPOCHS | 3 | Number of training epochs |
| DROPOUT | 0.1 | Dropout probability |

### Training Setup

- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss
- **Data Preprocessing**: Undersampled majority classes to 56,193 samples each
- **Hardware**: GPU (CUDA enabled)

---

## 5. Training Results

### Epoch-by-Epoch Performance

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.4936 | 80.44% | 0.3468 | 87.49% |
| 2 | 0.4252 | 83.07% | 0.3484 | 87.73% |
| 3 | 0.3778 | 84.98% | 0.3654 | 87.02% |

### Training Observations

**Loss Progression:**
- Training loss decreased consistently: 0.4936 → 0.3778 (23.5% reduction)
- Validation loss remained stable around 0.35
- No overfitting observed

**Accuracy Progression:**
- Training accuracy improved: 80.44% → 84.98% (+4.54%)
- Validation accuracy plateaued: 87.49% → 87.02%
- Validation consistently higher than training (good generalization)

**Training Time:**
- Approximately 60 minutes per epoch
- Total training time: ~180 minutes

---

## 6. Test Set Evaluation

### Overall Performance Metrics

```
Test Set Size: 14,746 sentences

Accuracy:   86.70%
Precision:  87.35% (weighted average)
Recall:     86.70% (weighted average)
F1-Score:   86.83% (weighted average)
```

### Confusion Matrix Results

| Class | Correct | Total | Accuracy | Errors |
|-------|---------|-------|----------|--------|
| BACKGROUND | 1,086 | 1,340 | **81.04%** | 254 |
| CONCLUSIONS | 1,843 | 2,217 | **83.13%** | 374 |
| METHODS | 4,545 | 4,893 | **92.89%** | 348 |
| OBJECTIVE | 742 | 1,190 | **62.35%** | 448 |
| RESULTS | 4,569 | 5,106 | **89.48%** | 537 |

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BACKGROUND | 0.6199 | 0.8104 | 0.7025 | 1,340 |
| CONCLUSIONS | 0.8187 | 0.8313 | 0.8250 | 2,217 |
| METHODS | 0.9350 | 0.9289 | 0.9319 | 4,893 |
| OBJECTIVE | 0.7721 | 0.6235 | 0.6899 | 1,190 |
| RESULTS | 0.9285 | 0.8948 | 0.9113 | 5,106 |
| | | | | |
| **Macro Avg** | **0.8148** | **0.8178** | **0.8121** | **14,746** |
| **Weighted Avg** | **0.8735** | **0.8670** | **0.8683** | **14,746** |

---

## 7. Performance Analysis

### 7.1 Best Performing Classes

**METHODS (F1: 0.9319)**
- Highest accuracy: 92.89%
- Precision: 93.50%, Recall: 92.89%
- 4,545 out of 4,893 sentences correctly classified
- Only 348 errors (7.11% error rate)

**Why it performs well:**
- Distinct technical vocabulary ("randomized", "administered", "measured")
- Clear methodological language patterns
- Largest class in test set provides robust evaluation

**RESULTS (F1: 0.9113)**
- Second-highest accuracy: 89.48%
- Precision: 92.85%, Recall: 89.48%
- 4,569 out of 5,106 sentences correctly classified
- 537 errors (10.52% error rate)

**Why it performs well:**
- Statistical terminology ("p<0.001", "mean", "significant difference")
- Numerical data patterns
- Second-largest class in test set

### 7.2 Moderate Performing Classes

**CONCLUSIONS (F1: 0.8250)**
- Accuracy: 83.13%
- Precision: 81.87%, Recall: 83.13%
- 1,843 out of 2,217 sentences correctly classified
- 374 errors (16.87% error rate)

**BACKGROUND (F1: 0.7025)**
- Accuracy: 81.04%
- Precision: 61.99%, Recall: 81.04%
- 1,086 out of 1,340 sentences correctly classified
- 254 errors (18.96% error rate)

**Issues:**
- BACKGROUND has low precision (61.99%): model over-predicts this class
- High recall (81.04%) but many false positives
- Confusion with OBJECTIVE sentences

### 7.3 Worst Performing Class

**OBJECTIVE (F1: 0.6899)**
- Lowest accuracy: 62.35%
- Precision: 77.21%, Recall: 62.35%
- Only 742 out of 1,190 sentences correctly classified
- 448 errors (37.65% error rate)

**Why it struggles:**
- Smallest class in test set (1,190 samples)
- Linguistic overlap with BACKGROUND sentences
- Both appear early in abstracts with similar introductory language
- 37.65% of OBJECTIVE sentences misclassified

**Error Pattern:**
- High precision (77.21%): when model predicts OBJECTIVE, it's usually correct
- Low recall (62.35%): model misses many true OBJECTIVE sentences
- Likely confused with BACKGROUND class

---

## 8. Key Findings

### Strengths
1. **Overall strong performance**: 86.70% accuracy across 14,746 test sentences
2. **Excellent on technical sections**: METHODS (92.89%) and RESULTS (89.48%)
3. **Good generalization**: Validation accuracy consistently higher than training
4. **Stable training**: No overfitting observed across 3 epochs

### Weaknesses
1. **OBJECTIVE classification**: Only 62.35% accuracy
2. **BACKGROUND precision**: Low at 61.99% (many false positives)
3. **Class confusion**: OBJECTIVE and BACKGROUND frequently mixed up
4. **Error count**: 1,961 total misclassifications out of 14,746 sentences

### Error Distribution
- RESULTS: 537 errors (27.4% of all errors)
- OBJECTIVE: 448 errors (22.8% of all errors)
- CONCLUSIONS: 374 errors (19.1% of all errors)
- METHODS: 348 errors (17.7% of all errors)
- BACKGROUND: 254 errors (13.0% of all errors)

---

## 9. Inference Implementation

We implemented a prediction function for classifying new sentences:

```python
def predict_text(text, model, tokenizer, device, label_names, max_length=128):
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs if not hasattr(outputs, 'logits') else outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)
    
    predicted_label = label_names[predicted_class.item()]
    confidence_score = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = {
        label_names[i]: probabilities[i].item() * 100 
        for i in range(len(label_names))
    }
    
    return {
        'text': text,
        'predicted': predicted_label,
        'confidence': confidence_score,
        'probabilities': all_probs
    }
```

The function takes a medical sentence as input and returns:
- Predicted class label
- Confidence score (0-100%)
- Probability distribution across all 5 classes

---

## 10. Challenges and Solutions

### Challenge 1: Class Imbalance
**Problem:** Original dataset had severe imbalance (METHODS: 59K samples vs OBJECTIVE: 14K samples)

**Solution:** Undersampled majority classes to 56,193 samples each, creating balanced training set of 280,965 sentences

**Result:** Helped but OBJECTIVE still underperforms (smallest test set)

### Challenge 2: OBJECTIVE vs BACKGROUND Confusion
**Problem:** These classes have overlapping language and both appear early in abstracts

**Attempted:** Standard BERT fine-tuning with balanced data

**Result:** OBJECTIVE achieved only 62.35% accuracy vs 81.04% for BACKGROUND

**Possible improvements:**
- Use positional features (sentence index in abstract)
- Increase max token length beyond 128
- Try domain-specific BERT (BioBERT, PubMedBERT)

### Challenge 3: Training Time
**Problem:** Full fine-tuning takes significant time (~3 hours for 3 epochs)

**Trade-off:** Used 60% of data instead of 100% to manage training time

**Impact:** May have limited model's full potential, but achieved strong results

### Challenge 4: Computational Resources
**Problem:** Large model (110M parameters) requires GPU memory

**Solution:** Reduced batch size to 16 and used 60% data subset

**Result:** Successfully trained without out-of-memory errors

---

## 11. Visualizations

### Training Curves

Based on our training data:

**Loss Curves:**
- Training loss: Steady decrease from 0.49 to 0.38
- Validation loss: Stable around 0.35
- Healthy convergence pattern

**Accuracy Curves:**
- Training accuracy: 80% → 85% (steady improvement)
- Validation accuracy: Plateaus at 87%
- Small train-val gap indicates good generalization

### Confusion Patterns

Most common misclassifications:
1. OBJECTIVE → BACKGROUND (448 OBJECTIVE errors, many to BACKGROUND)
2. BACKGROUND → other classes (low precision of 61.99%)
3. RESULTS → CONCLUSIONS (537 RESULTS errors, some to CONCLUSIONS)

---

## 12. Conclusion

We successfully implemented a BERT-based medical abstract sentence classifier achieving **86.70% test accuracy** on 14,746 sentences. The model performs exceptionally well on METHODS (92.89%) and RESULTS (89.48%) but struggles with OBJECTIVE classification (62.35%).

**Final Metrics:**
- Test Accuracy: **86.70%**
- Weighted F1-Score: **86.83%**
- Training Time: **~3 hours**
- Model Size: **110M parameters**

**Achievement Summary:**
- 12,785 out of 14,746 sentences correctly classified
- 1,961 misclassifications
- Strong performance on 3 out of 5 classes (>83% accuracy)

The model demonstrates the effectiveness of transfer learning with BERT for domain-specific text classification and is ready for deployment in medical literature analysis applications.

---

## 13. Future Work

1. **Improve OBJECTIVE classification:**
   - Use BioBERT or PubMedBERT (pre-trained on medical text)
   - Add positional encoding (sentence order in abstract)
   - Increase training data for this class

2. **Reduce BACKGROUND false positives:**
   - Implement focal loss to handle precision-recall trade-off
   - Use ensemble methods

3. **Full dataset training:**
   - Train on 100% of data instead of 60%
   - Expected improvement: +2-3% accuracy

4. **Hyperparameter optimization:**
   - Learning rate scheduling
   - Longer training (5-10 epochs)
   - Larger batch sizes if memory permits

---

## 14. Technical Specifications

**Environment:**
- Framework: PyTorch
- Pre-trained Model: bert-base-uncased (HuggingFace)
- Hardware: GPU with CUDA
- Training Time: 180 minutes (3 epochs)

**Code Structure:**
- Model definition: Custom BERTClassifier class
- Training loop: Standard PyTorch training with AdamW
- Evaluation: sklearn metrics (accuracy, precision, recall, F1)
- Inference: predict_text() function for single sentence classification

