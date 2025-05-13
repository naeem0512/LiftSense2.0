# Baseline LSTM Model (v1.0)
- **Test Accuracy**: 93.0%
- **Val Loss**: 0.1127
- **Key Features**:
  - Rep Speed
  - Force Output
  - Heart Rate
  - Grip Strength
- **Architecture**:
  ```python
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                Output Shape              Param #   
  =================================================================
  lstm (LSTM)                 (None, 128)               67584     
  _________________________________________________________________
  dense (Dense)               (None, 3)                 387       
  =================================================================

---

### **📌 Phase 1: Upgrade Pipeline (Without Losing Original)**
#### **1. New Hybrid Model Directory**
```bash
mkdir -p src/hybrid_model
cp src/model_training.py src/hybrid_model/lstm_baseline.py  # Archive original