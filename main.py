import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd # Excel çıktısı için gerekli
import numpy as np
import os

# Kendi dosyalarımız
from data.data_loading import get_kvasir_data, get_train_val_split
from local_datasets.dataset import KvasirHFDataset
from models.model import SimpleVQA

def tokenize_question(question, vocab, max_len=15):
    tokens = question.lower().split()
    token_ids = [vocab.get(t, 0) for t in tokens]
    if len(token_ids) < max_len:
        token_ids += [0] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return torch.tensor(token_ids, dtype=torch.long)

def plot_confusion_matrix(y_true, y_pred, classes):
    # Sınıf sayısı çok fazla olduğu için grafiği büyütüp kaydediyoruz
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    # annot=False yaptık çünkü 400 sınıfta sayılar üst üste biner, okunmaz
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues') 
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('ResNet + LSTM Confusion Matrix')
    
    # Resmi kaydet
    plt.savefig("resnetLSTM_matrix.png", dpi=300)
    print("📊 Karmaşıklık matrisi 'resnetLSTM_matrix.png' olarak kaydedildi.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- ResNet + LSTM Modeli Başlatılıyor ---")
    print(f"Kullanılan Cihaz: {device}")

    # 1. VERİ YÜKLEME VE SÖZLÜK (Optimize Edilmiş Hali)
    raw_data = get_kvasir_data()
    train_data, val_data = get_train_val_split(raw_data, split_ratio=0.8) 
    
    print("Sözlükler oluşturuluyor...")
    
    answers_list = train_data['answer'] 
    questions_list = train_data['question'] 
    
    all_answers = sorted(list(set(str(ans).lower() for ans in answers_list)))
    answer_map = {ans: i for i, ans in enumerate(all_answers)}
    
    all_text = " ".join([str(q).lower() for q in questions_list])
    vocab = {word: i+1 for i, word in enumerate(set(all_text.split()))}
    vocab['<unk>'] = 0
    
    print(f"-> İşlem Tamam! Sınıf Sayısı: {len(answer_map)}")

    # 2. DATASET AYARLARI
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['answer'] for item in batch])
        questions = torch.stack([tokenize_question(item['question'], vocab) for item in batch])
        return images, questions, labels

    train_ds = KvasirHFDataset(train_data, answer_map, transform)
    val_ds = KvasirHFDataset(val_data, answer_map, transform)

    # Senin bilgisayarın için Batch Size 8 kalmalı
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 3. MODEL KURULUMU
    model = SimpleVQA(num_classes=len(answer_map), vocab_size=len(vocab)+1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- AKILLI KAYDETME SİSTEMİ ---
    MODEL_PATH = "resnet_lstm_model.pth"

    if os.path.exists(MODEL_PATH):
        print(f"\n✅ Kayıtlı model bulundu ({MODEL_PATH}). Eğitim ATLANIYOR...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("\n--- Eğitim Başlıyor ---")
        EPOCHS = 3 
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for imgs, quests, labels in loop:
                imgs, quests, labels = imgs.to(device), quests.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs, quests)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            print(f"Epoch {epoch+1} Bitti. Ort. Loss: {total_loss/len(train_loader):.4f}")
        
        # Eğitimi Kaydet
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Model başarıyla kaydedildi: {MODEL_PATH}")

    # 4. TEST VE RAPORLAMA
    print("\n--- Test Raporu Hazırlanıyor ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, quests, labels in tqdm(val_loader, desc="Tahminler"):
            imgs, quests, labels = imgs.to(device), quests.to(device), labels.to(device)
            outputs = model(imgs, quests)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ID ve İsim Listeleri
    label_ids = list(answer_map.values())
    label_names = list(answer_map.keys())

    # A) TXT RAPORU
    report_text = classification_report(all_labels, all_preds, 
                                        labels=label_ids, 
                                        target_names=label_names, 
                                        zero_division=0)
    with open("resnet_rapor.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("✅ 'resnetLSTM_rapor.txt' kaydedildi.")

    # B) EXCEL/CSV RAPORU
    report_dict = classification_report(all_labels, all_preds, 
                                        labels=label_ids, 
                                        target_names=label_names, 
                                        zero_division=0,
                                        output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv("resnetLSTM_analiz.csv")
    print("✅ 'resnetLSTM_analiz.csv' kaydedildi (Excel formatı).")
    
    # C) GRAFİK
    plot_confusion_matrix(all_labels, all_preds, label_names)

if __name__ == "__main__":
    main()