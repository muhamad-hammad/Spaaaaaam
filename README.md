# 🛡️ Spam Detector App

A simple and interactive **Spam Detection** web application built using Streamlit and a trained **Multinomial Naive Bayes** model.

---

## 🚀 Features

- 📝 Input any message to check for spam
- 🔄 Text preprocessing (tokenization, stopwords removal, stemming)
- 🤖 Prediction using a trained machine learning model
- 🧠 Displays whether message is **Spam** or **Not Spam**
- 📜 History of your last 5 checks

---

## 📁 Project Structure

| File               | Description                                 |
|--------------------|---------------------------------------------|
| `app.py`           | Streamlit app file                          |
| `main.ipynb`       | Jupyter notebook for training/testing model |
| `MultinomialNB.pkl`| Trained Naive Bayes model (required)        |
| `Vectorizer.pkl`   | Trained text vectorizer (required)          |

---

## 💻 How to Run

### 🔽 Step 1: Clone from GitHub

```bash
git clone https://github.com/YOUR_USERNAME/spam-detector-app.git
cd spam-detector-app
```

> Replace `YOUR_USERNAME` with your GitHub username.

---

### ⚙️ Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install:

```bash
pip install streamlit nltk scikit-learn numpy
```

And download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

### ▶️ Step 3: Run the App

```bash
streamlit run app.py
```

The app will open in your default browser.

---

## 💡 Example Use

**Input:**  
`"Congratulations! You won a lottery worth $1000. Click here!"`  
**Output:**  
🚨 **SPAM**

**Input:**  
`"Hey, can we meet for coffee tomorrow?"`  
**Output:**  
✅ **Not Spam**

---

## 🧰 Tech Stack

- Python
- Streamlit
- NLTK
- Scikit-learn

---

## 👤 Author

**Muhammad Hammad (i230544)**

---

## 📜 License

This project is intended for educational use.