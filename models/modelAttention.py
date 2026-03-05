import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisualAttentionVQA(nn.Module):
    def __init__(self, num_classes, vocab_size=10000, embed_dim=256, hidden_dim=512):
        super(VisualAttentionVQA, self).__init__()
        
        # 1. GÖRÜNTÜ ÖZELLİKLERİ (ResNet18 - Spatial)
        # Dikkat! Burada 'pooling' yapmıyoruz, 7x7 ızgarayı koruyoruz.
        resnet = models.resnet18(pretrained=True)
        # Son iki katmanı (AvgPool ve FC) atıyoruz
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2]) 
        # ResNet18 çıktısı: [Batch, 512, 7, 7]
        
        # Resim boyutunu bizim hidden_dim'e eşitleyelim
        self.cnn_fc = nn.Conv2d(512, hidden_dim, kernel_size=1) 
        
        # 2. SORU ÖZELLİKLERİ (LSTM)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 3. ATTENTION MEKANİZMASI (DİKKAT KATMANI)
        # Formül: Softmax( W * tanh( Wv*Image + Wq*Question ) )
        self.w_img = nn.Linear(hidden_dim, hidden_dim)
        self.w_ques = nn.Linear(hidden_dim, hidden_dim)
        self.w_att = nn.Linear(hidden_dim, 1) # Skora çevir
        
        # 4. SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, question_indices):
        # --- A. RESİM İŞLEME ---
        # ResNet Çıktısı: [Batch, 512, 7, 7]
        img_feat = self.resnet_features(images)
        
        # Kanal sayısını 512'den hidden_dim'e düşür: [Batch, hidden, 7, 7]
        img_feat = self.cnn_fc(img_feat)
        
        # Düzleştirme (Flatten Spatial): [Batch, 49, hidden] (49 = 7x7)
        b, c, h, w = img_feat.size()
        img_feat = img_feat.view(b, c, -1).permute(0, 2, 1) 
        
        # --- B. SORU İŞLEME ---
        embeds = self.embedding(question_indices)
        # LSTM Çıktısı: output, (hidden, cell)
        # Biz son gizli durumu (hidden state) alıyoruz: [Batch, hidden]
        _, (hidden_state, _) = self.lstm(embeds)
        ques_feat = hidden_state[-1]
        
        # --- C. DİKKAT (ATTENTION) HESAPLAMA ---
        # Amaç: 49 tane bölgeden hangisi soruyla alakalı?
        
        # Resim özelliklerini genişlet
        img_proj = self.w_img(img_feat) # [Batch, 49, hidden]
        
        # Soru özelliklerini genişlet (Her piksel için tekrarla)
        ques_proj = self.w_ques(ques_feat).unsqueeze(1).expand_as(img_proj) # [Batch, 49, hidden]
        
        # Birleştir ve Puanla
        att_score = torch.tanh(img_proj + ques_proj) # [Batch, 49, hidden]
        att_alpha = F.softmax(self.w_att(att_score), dim=1) # [Batch, 49, 1] (0 ile 1 arası ağırlıklar)
        
        # --- D. AĞIRLIKLI BİRLEŞTİRME ---
        # Resim bölgelerini dikkat ağırlıklarıyla çarpıp topla
        # (Örn: Polip olan bölge 0.9, diğer yerler 0.01 ile çarpılır)
        weighted_img = (img_feat * att_alpha).sum(dim=1) # [Batch, hidden]
        
        # --- E. SINIFLANDIRMA ---
        # Sadece resmin önemli kısmı + Soru bilgisi
        # (Bazı modellerde weighted_img + ques_feat birleştirilir, burada basit tutuyoruz)
        combined = weighted_img + ques_feat
        output = self.classifier(combined)
        
        return output