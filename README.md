# -Customer-Segmentation-using-FNN-Feedforward-Neural-Network-
# 🧠 Customer Segmentation using FNN (Feedforward Neural Network)

This project applies **KMeans clustering** to segment mall customers and then uses a **Feedforward Neural Network (FNN)** to classify customers into their respective segments.

## 📌 Dataset

**Mall Customers Dataset**  
- Source: [Kaggle - Mall Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)
- Features used:  
  - Annual Income (k$)  
  - Spending Score (1–100)

## 🧪 Objective

- Segment customers based on spending patterns and income using **unsupervised KMeans**.
- Train a neural network to classify customers into these segments (simulated supervision).

## ⚙️ Technologies Used

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn (for clustering and preprocessing)
- TensorFlow / Keras (for the neural network)

## 📊 Steps

1. **Data Preprocessing**
   - Load and clean data
   - Feature scaling using `StandardScaler`
2. **Customer Segmentation**
   - Apply KMeans clustering with 5 clusters
   - Use cluster labels as pseudo-targets
3. **Model Training**
   - Build a neural network with:
     - 2 Hidden Layers (ReLU activation)
     - Output Layer with Softmax for 5 classes
   - Train with `categorical_crossentropy` loss
4. **Evaluation**
   - Accuracy and loss curves plotted using Matplotlib

## 🔍 Results

- The FNN achieved good classification accuracy on cluster labels.
- Visual analysis of training/validation accuracy and loss is included.

## 🧠 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn tensorflow

# Run the script
python customer_segmentation_fnn.py
````

## 📌 Note

This project treats **clustering labels as supervised targets** for learning purposes. This is not typical for real segmentation tasks, which are usually unsupervised.

---

### 🔗 Related Topics

* Clustering
* Unsupervised Learning
* Neural Networks
* Customer Analytics

## 👩‍💻 Author
Speranza Deejoe 
