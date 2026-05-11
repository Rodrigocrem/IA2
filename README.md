# EmotiScan - Face Emotion AI Dashboard

Dashboard interactivo para clasificacion de expresiones faciales con CNN y MobileNetV2.

**Practica 2 - Grupo C7**

## Arquitectura

```
emotion-dashboard/
├── backend/          # FastAPI
│   ├── main.py       # Endpoints: /predict, /metrics/*, /config
│   ├── requirements.txt
│   └── models/       # .keras / .h5 aqui
└── frontend/
    └── index.html    # Dashboard HTML/JS/CSS standalone
```

## Setup local

```bash
# Backend
cd backend
pip install -r requirements.txt
# Copiar modelos entrenados (opcional):
# cp best_model.keras models/best_cnn_model.keras
# cp best_tl_model.keras models/best_transfer_model.keras
uvicorn main:app --reload --port 8000

# Frontend (otra terminal)
cd frontend && python3 -m http.server 3000
# Abre http://localhost:3000
```

## Endpoints

| Endpoint | Descripcion |
|----------|-------------|
| POST /api/predict | Prediccion con CNN o MobileNetV2 |
| GET /api/metrics/history | Curvas de entrenamiento |
| GET /api/metrics/confusion | Matriz de confusion |
| GET /api/metrics/per-class | Precision/Recall/F1 por clase |
| GET /api/metrics/dataset | Distribucion del dataset |
| GET /api/metrics/comparison | CNN vs Transfer Learning |
| GET /api/config | Hiperparametros |

Docs en: http://localhost:8000/docs

## Modelos

| Modelo | Val Accuracy |
|--------|-------------|
| CNN personalizada | 58% |
| MobileNetV2 Transfer | 77% |

Dataset: Human Face Emotions (Kaggle) - 7834 imgs, 5 clases
