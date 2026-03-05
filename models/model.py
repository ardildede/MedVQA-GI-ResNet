import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel

class CoAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CoAttentionLayer, self).__init__()
        # Görüntü ve metin arasındaki etkileşimi kuran çok kafalı dikkat mekanizması
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # query: Görüntü özellikleri, key/value: Metin özellikleri
        attn_output, _ = self.multihead_attn(query, key, value)
        return self.norm(query + self.dropout(attn_output))

class ResNet_BERT_CoAttention_VQA(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_BERT_CoAttention_VQA, self).__init__()
        
        # 1. GÖRÜNTÜ: ResNet50 (Uzamsal özellikleri korumak için son iki katman atıldı)
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2]) # Çıktı: [Batch, 2048, 7, 7]
        
        # ResNet'in 2048 kanalını BERT'in 768 boyutuna indirgiyoruz
        self.img_projection = nn.Conv2d(2048, 768, kernel_size=1)
        
        # 2. METİN: DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 3. FUSION: Co-Attention
        self.co_attention = CoAttentionLayer(embed_dim=768)
        
        # 4. SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Görüntü Özellikleri: [B, 2048, 7, 7] -> [B, 768, 7, 7]
        img_feats = self.resnet_features(images)
        img_feats = self.img_projection(img_feats)
        
        # Görüntüyü sekans haline getir: [B, 768, 49] -> [B, 49, 768] (49 bölge/token)
        b, c, h, w = img_feats.shape
        img_seq = img_feats.view(b, c, h*w).permute(0, 2, 1)
        
        # Metin Özellikleri: [B, Seq_Len, 768]
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = bert_out.last_hidden_state
        
        # Co-Attention: Görüntü bölgeleri metin kelimelerine bakıyor
        fused_seq = self.co_attention(img_seq, text_seq, text_seq)
        
        # Bölgelerin ortalamasını alarak global bir temsil oluştur
        fused_global = fused_seq.mean(dim=1)
        
        # Sınıflandırma
        return self.classifier(fused_global)