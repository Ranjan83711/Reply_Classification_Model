import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import matplotlib.pyplot as plt
import os

# Step 1: Load and Preprocess Dataset
def load_and_preprocess(data_path='reply_classification_dataset.csv'):
    df = pd.read_csv(data_path)
    
    # Handle missing values
    df = df.dropna(subset=['reply', 'label'])
    
    # Clean text: remove special chars, lowercase
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
        text = text.lower().strip()
        return text
    
    df['reply'] = df['reply'].apply(clean_text)
    
    # Map labels to integers for transformers
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    df['label_int'] = df['label'].map(label_map)
    
    # Drop rows where label_int is NaN (unmapped labels)
    df = df.dropna(subset=['label_int'])
    df['label_int'] = df['label_int'].astype(int)
    
    return df

# Step 2: Train Baseline Model (Logistic Regression with TF-IDF)
def train_baseline(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return acc, f1

# Step 3: Fine-tune DistilBERT
def train_distilbert(df):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Tokenize
    train_encodings = tokenizer(train_df['reply'].tolist(), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_df['reply'].tolist(), truncation=True, padding=True, max_length=128)
    
    class ReplyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = ReplyDataset(train_encodings, [int(x) for x in train_df['label_int'].tolist()])
    test_dataset = ReplyDataset(test_encodings, [int(x) for x in test_df['label_int'].tolist()])
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='epoch',  # <-- change here
        save_strategy='epoch',
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    trainer.train()
    
    # Evaluate
    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(-1)
    y_test = test_df['label_int']
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_model')
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Model and tokenizer saved successfully.")
    
    return acc, f1

# Step 4: Compare and Explain
def compare_models(baseline_acc, baseline_f1, bert_acc, bert_f1):
    print(f"Baseline (Logistic Regression): Accuracy={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    print(f"DistilBERT: Accuracy={bert_acc:.4f}, F1={bert_f1:.4f}")
    
    # Plot comparison
    models = ['Logistic Regression', 'DistilBERT']
    accs = [baseline_acc, bert_acc]
    f1s = [baseline_f1, bert_f1]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, accs, width=0.4, label='Accuracy', align='center', color='#1f77b4')
    plt.bar(models, f1s, width=0.4, label='F1 Score', align='edge', color='#ff7f0e')
    plt.legend()
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    # Explanation
    print("\nExplanation for Production Choice:")
    print("I recommend using DistilBERT in production because it captures semantic context and nuances in text better than TF-IDF-based models, leading to higher accuracy and robustness on varied replies. Despite being computationally heavier, DistilBERT is lightweight compared to larger transformers, making it suitable for deployment. The baseline model (Logistic Regression) is faster for very large-scale inference but less effective for complex language understanding.")

# Main Execution
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_model'), exist_ok=True)
    
    df = load_and_preprocess()
    print(f"Loaded dataset with {len(df)} rows.")
    
    # Split for baseline
    X_train, X_test, y_train, y_test = train_test_split(df['reply'], df['label'], test_size=0.2, random_state=42)
    
    baseline_acc, baseline_f1 = train_baseline(X_train, X_test, y_train, y_test)
    bert_acc, bert_f1 = train_distilbert(df)
    
    compare_models(baseline_acc, baseline_f1, bert_acc, bert_f1)