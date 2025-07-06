# ornek

Bu repo, Kaggle'da yaygin olarak kullanilan Titanik veri seti üzerinde basit bir makine öğrenmesi çalışması içerir. 

`Titanic` adlı Jupyter defteri veri görselleştirme ve tahmin modelleri üerinde durur. 
Ayrıca yeni eklenen `titanic_logistic_regression.py` betiği, öğretici amaçlı olarak 
scikit-learn kullanarak basit bir lojistik regresyon modeli eğitir.

## Gereksinimler

Python 3 ile birlikte aşağıdaki paketler gereklidir:

```
pip install pandas seaborn scikit-learn
```

## Betiği çalıştırmak

```
python titanic_logistic_regression.py
```

Betiğ, Seaborn kütüphanesinden Titanik veri setini yükler, ön işleme 
yapar ve test doğruluğunu ekrana yazar.
