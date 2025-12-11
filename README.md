"# deeplearning" streamlit 
<img width="1902" height="543" alt="image" src="https://github.com/user-attachments/assets/dd01c346-4fa3-427c-965e-813cf8b03b22" />
---

# 📄 **Final Project Report**

## **EuroSAT & CIFAR-10 Classification using Transfer Learning and Web Deployment**

---

## **1. Introduction**

This project focuses on developing, evaluating, and deploying two image classification models trained on two different datasets: **EuroSAT** (satellite land-cover images) and **CIFAR-10** (natural object images).
The aim is to understand the impact of **domain shift**, apply **transfer learning**, and deploy a real-time inference system using **Streamlit**.

The final system allows users to upload an image and choose which model (EuroSAT or CIFAR-10) will perform the prediction.

---

## **2. Datasets Description**

### **2.1 EuroSAT Dataset**

* Contains **27,000+ multispectral satellite images** from sentinel-2.
* RGB version used in this project.
* Image size: **64×64 px**.
* Includes **10 land-use classes**:

  * Forest, River, Pasture, Residential, Industrial, Highway, Farmland, SeaLake, HerbaceousVegetation, PermanentCrop.
* Represents a **remote sensing** domain.

### **2.2 CIFAR-10 Dataset**

* A classic computer vision dataset with **60,000 natural images**.
* Image size: **32×32 px**.
* Contains **10 classes**:

  * Cat, Dog, Car, Truck, Airplane, Bird, Deer, Frog, Ship, Horse.
* Represents the **natural-image domain**.

---

## **3. Data Preprocessing**

Both datasets underwent the same processing pipeline using `fastai`:

### **3.1 Image Resizing**

All images were resized to a consistent resolution (e.g., 64×64 px) to match the CNN input requirement.

### **3.2 Normalization**

Images were normalized using dataset statistics to accelerate convergence.

### **3.3 Data Augmentation**

The following transformations were applied:

* Random horizontal flip
* Random rotate
* Lighting/backlighting adjustments
* Random crop

This improved model generalization.

---

## **4. Model Development**

Two models were developed based on **ResNet architecture** using the `fastai` library.

---

### **4.1 EuroSAT Classification Model**

* Trained using fastai’s `cnn_learner`.
* Achieved good accuracy on satellite land-cover images.
* Saved as **`eurosat.pkl`** for inference.

---

### **4.2 CIFAR-10 Classification Model**

* Trained using transfer learning with ResNet50.
* Fine-tuned on CIFAR-10 dataset.
* Saved as **`cifar.pkl`**.

---

## **5. Transfer Learning & Domain Shift**

### **5.1 Domain Shift Explanation**

A model trained on one domain (e.g., satellite images) **fails** when tested on a different domain (natural images), because the features are very different.

### **5.2 Experiment Observations**

* When testing **EuroSAT model** on CIFAR images → predictions become incorrect (because the domain is completely different).
* When testing **CIFAR model** on EuroSAT images → similar failure occurs.

This illustrates that deep learning models strongly rely on domain-specific patterns.

### **5.3 Transfer Learning Benefit**

Instead of training from scratch:

* CIFAR model benefits from pretrained ResNet weights.
* EuroSAT model could also be adapted to benefit from pretrained ImageNet weights.

This improves convergence and performance.

---

## **6. Model Inference (Prediction Pipeline)**

Streamlit cannot use fastai’s built-in `.predict()` method due to progress bar callbacks.
Therefore, a **custom prediction function** was written:

### **manual_predict( )**

* Builds a one-image test loader
* Extracts the batch tensor
* Feeds it manually into the model (`learn.model(xb)`)
* Applies Softmax to obtain probabilities
* Returns:

  * Predicted class name
  * Class index
  * Probability vector

This ensures stable inference in Streamlit without training behavior.

---

## **7. Web Deployment Using Streamlit**

A full web app (`app.py`) was built with the following features:

### ✔ Upload an image

### ✔ Choose between:

* EuroSAT Model
* CIFAR Model

### ✔ Display:

* Predicted class
* Confidence percentage
* Detailed class probability distribution

### ✔ Backend:

* Models loaded from `.pkl` files
* Prediction done using `manual_predict()`
* Ngrok tunneling used to expose Streamlit from Google Colab to a public link

### **Deployment Command (Colab):**

```bash
!streamlit run app.py --server.port 8501 &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## **8. Results and Discussion**

### **EuroSAT Model Performance**

* Performs well on satellite imagery.
* Poor results when tested on natural images (correct behavior due to domain shift).

### **CIFAR-10 Model Performance**

* Performs well on natural objects.
* Fails on satellite images—again confirming domain shift.

### **Important Insight**

> **No model works correctly outside the domain it was trained on**, unless domain adaptation or fine-tuning is applied.

This highlights the importance of dataset choice and the limits of deep learning generalization.

---

## **9. Final Deliverables**

* ✔ EuroSAT model – `eurosat.pkl`
* ✔ CIFAR model – `cifar.pkl`
* ✔ Streamlit App – `app.py`
* ✔ Ngrok deployment URL
* ✔ Project notebook (workflow, training, results, visualizations)

---

## **10. Conclusion**

This project demonstrates a complete deep learning pipeline including:

* Dataset preparation
* Model training and fine-tuning
* Domain shift understanding
* Transfer learning
* Custom inference pipeline
* Full real-time web deployment


## 🎯 جاهز أعملهولك PDF أو Word؟

لو تحب:

✔ أحوّل التقرير لملف Word جاهز
✔ أو PDF جاهز للطباعة
✔ أو PowerPoint Presentation كاملة

قولّي وأنا أعمله فورًا ❤️🚀

