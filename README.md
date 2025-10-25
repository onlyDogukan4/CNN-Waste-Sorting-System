# CNN-Waste-Sorting-System


1- 🚀 Proje Genel Bakış ve Portföy Amacı

Bu proje, görüntü işleme (Computer Vision) ve Derin Öğrenme (Deep Learning) alanındaki yetkinliğimi sergilemek amacıyla, altı farklı atık türünü (**Karton, Cam, Metal, Kağıt, Plastik, Çöp**) yüksek doğrulukla otomatik olarak sınıflandırmak için geliştirilmiştir.

Amacım, bir staj veya kariyer fırsatı için başvurduğum şirketlere, **uçtan uca bir makine öğrenimi projesi yönetme, model analizi yapma ve sonuçları profesyonelce raporlama** becerisine sahip olduğumu göstermektir.

### Teknik Özet ve Elde Edilen Başarı
* **Model Mimarisi:** Custom 2D CNN (Convolutional Neural Network)
* **Final Test Doğruluğu:** **[% [0.8065] ]**
* **Kullanılan Teknikler:** Data Augmentation, Keras Callbacks (Early Stopping, Model Checkpointing).
* **Kullanılan Kütüphaneler:** TensorFlow/Keras, OpenCV, Scikit-learn.

[![](https://img.shields.io/badge/Notebook-Colab-yellow.svg)](notebooks/Garbage_Classification_Training.ipynb) 

---

2- ## Model Performansı ve Analiz

Modelimizin eğitim süreçleri ve nihai performansı, detaylı görsel analizlerle incelenmiştir.

### 2.1. Eğitim Tarihçesi (Loss & Accuracy)

(results/accuracy_history.png) | (results/loss_history.png) |

** Analiz Yorumu :**
"Doğrulama Kaybı grafiği, yaklaşık **[70.]** epoch'tan sonra yükselişe geçmiştir. Bu, modelin eğitim verisini ezberlemeye başladığı (overfitting) noktadır. **`EarlyStopping` geri çağrısı** kullanılarak, eğitim bu kritik noktada durdurulmuş ve en iyi genelleme yeteneğine sahip ağırlıklar Drive'a kaydedilmiştir. Bu mekanizma, modelin stabilitesini ve gerçek dünya verilerine uyumunu sağlamıştır."

---


### 2.2. Sınıf Bazlı Detaylı Metrikler

Modelin her bir sınıf üzerindeki performansını **Precision**, **Recall** ve **F1 Skoru** metrikleri ile değerlendirdik. Bu detaylı analiz, modelin zayıf ve güçlü olduğu alanları gösterir.

#### Karışıklık Matrisi (Confusion Matrix)
(results/confusion_matrix.png)

#### Sınıflandırma Raporu Isı Haritası
(results/classification_report_heatmap.png)

**Analiz Yorumu :**
"Isı haritası, modelin **[kağıt]** gibi görsel olarak belirgin sınıflarda yüksek F1 skorları elde ettiğini göstermektedir. Ancak, 'Trash' gibi karmaşık sınıflarda nispeten düşük skorlar almıştır. Bu, gelecekteki iyileştirme adımları için **veri artırma stratejileri** veya **Transfer Öğrenimi** gibi daha gelişmiş tekniklere odaklanılması gerektiğini işaret etmektedir."

---

### 2.3. Modelin Hata ve Başarı Görselleştirmesi 

Bir modelin genelleme yeteneğini anlamak için hatalarını analiz etmek kritiktir.

| Modelin Yanlış Sınıflandırdığı Örnekler (Zayıf Yönler) | Modelin Doğru Sınıflandırdığı Örnekler (Güçlü Yönler) |

| (results/misclassified_samples.png) | (results/correctly_classified_samples.png) |

**Analiz Yorumu :**
"Yanlış sınıflandırılmış görseller, modelin özellikle **görsel benzerlik** nedeniyle zorlandığını göstermiştir (örn: **[Buruşturulmuş kağıt attığımda trash dedi]**). Bu analiz, modelin limitlerini anlamak ve veri kalitesini artırmak için yol haritası belirlemek adına önemli bir adımdır."

---

## Kurulum ve Çalıştırma

Projenin tekrarlanabilirliğini sağlamak ve staj yaptığım/başvurduğum ekibin kolayca incelemesi için gerekli adımlar aşağıdadır.

### 1. Bağımlılıklar
Gerekli tüm kütüphaneler ve versiyonları `requirements.txt` dosyasında listelenmiştir.

```bash
pip install -r requirements.txt
