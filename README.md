# 🚀 **Text-Based Question Answering System** 📚  

### **Predict answers to questions from a given story using Deep Learning!**

---

## 🌟 **Overview**  
This project implements a **text-based question-answering (QA) system** using a **Neural Network** built with **Keras** and **TensorFlow**. The system processes story-question-answer triplets, applies an **attention mechanism**, and leverages **LSTMs** to predict answers like "yes" or "no" based on the context provided in a story.  

With this model, you can:  
✅ Train on custom datasets of stories and questions.  
✅ Predict answers for your own stories and questions.  
✅ Visualize model accuracy during training.  

---

## 🛠️ **Features**  
- **End-to-End Workflow**: From data preprocessing to prediction.  
- **Neural Network Architecture**:  
   - **Embeddings**: For word representation.  
   - **Attention Mechanism**: To capture relationships between story and question.  
   - **LSTM Layers**: To process sequences effectively.  
   - **Softmax Output**: To predict answers.  
- **Custom Input Support**: Test the model with your own story-question pairs.  
- **Visualization**: View training and validation accuracy over epochs.  
- **Model Persistence**: Save and reuse the trained model.  

---

## 📊 **Model Architecture**  

```
Input Story --> Embedding --> Attention Mechanism --> LSTM --> Dense Layer --> Answer Prediction
Input Question --> Embedding --> Attention Mechanism --> LSTM --> Dense Layer --> Answer Prediction
```

The model intelligently maps relationships between **story** and **question** to output a **predicted answer**.

---

## ⚙️ **Setup & Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/qa-system.git
cd qa-system
```

### **2. Install Dependencies**  
Make sure Python and pip are installed. Then run:  
```bash
pip install tensorflow numpy matplotlib
```

### **3. Add Your Dataset**  
Place `train_qa.txt` and `test_qa.txt` in the project directory.  

---

## 🚀 **How to Use**  

### **1. Train the Model**  
Run the script to train the model:  
```bash
python script_name.py
```

### **2. Visualize Training Results**  
The training and validation accuracy will be plotted for performance analysis.  

### **3. Custom Predictions**  
Test the model with your own story and question:  

```python
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_question = "Is the football in the garden ?"

# Run prediction
```

**Output:**  
```
Predicted Answer: yes  
Confidence Score: 0.95
```

---

## 📈 **Training Details**  
- **Optimizer**: RMSProp  
- **Loss Function**: Categorical Cross-Entropy  
- **Epochs**: 120  
- **Batch Size**: 32  

---

## 🎯 **Results**  
- Achieved high accuracy on the validation set.  
- Model generalizes well to unseen story-question pairs.  
- Outputs a probability distribution over the vocabulary for predictions.  

---

## 🖼️ **Visualizations**  
Training and validation accuracy are plotted to evaluate model performance:  

![Accuracy Plot](https://via.placeholder.com/600x300.png?text=Training+vs+Validation+Accuracy)

---

## 📚 **Dataset**  
The dataset consists of:  
- **Stories**: Context for the question.  
- **Questions**: Queries based on the story.  
- **Answers**: Labels (e.g., "yes" or "no").  

---

## 💡 **Why This Project?**  
This project showcases the power of **Neural Networks** in solving natural language processing tasks, specifically question answering. It demonstrates:  
- Efficient data preprocessing using tokenization and padding.  
- Building a model with **attention** to focus on key parts of the input.  
- Training and evaluating deep learning models for text-based tasks.  

---

## 📥 **Future Improvements**  
- Extend to multi-word answers.  
- Use advanced NLP techniques like transformers.  
- Add support for larger, more complex datasets.

---

## 🔗 **Dependencies**  
- TensorFlow  
- NumPy  
- Matplotlib  

---

## 📜 **License**  
This project is licensed under the **MIT License**.  

---

## 🙌 **Contributions**  
Feel free to fork, improve, and create pull requests. Contributions are welcome!  

---

## 🧑‍💻 **Author**  
**Harshit Srivastava**  
- GitHub: https://github.com/DevvVision

---

**⭐ If you found this project useful, don't forget to give it a star! ⭐**

