# 🧠 Handwritten Digit Recognition & Generation using GANs

A full-stack machine learning project that combines **Convolutional Neural Networks (CNNs)** for digit recognition and **Generative Adversarial Networks (GANs)** for synthetic digit generation.

This project allows users to:

* ✍️ Draw handwritten digits and get predictions
* 🎨 Generate realistic handwritten digits using GANs
* 📊 Visualize model performance and generated samples

---

## 🚀 Features

* 🔢 **Digit Recognition** using CNN (MNIST dataset)
* 🎨 **Digit Generation** using GAN
* 🌐 **Web Interface** with drawing canvas
* ⚡ **FastAPI Backend** for model inference
* 📓 **Jupyter Notebooks** for experimentation and visualization
* 📈 Training visualizations and generated samples

---

## 🧠 Project Architecture

```
User Input (Canvas)
        ↓
Frontend (HTML/JS)
        ↓
FastAPI Backend
        ↓
-------------------------
|  CNN Classifier       | → Predict Digit
|  GAN Generator        | → Generate Digits
-------------------------
```

---

## 📁 Project Structure

```
digit-gan-app/
│
├── notebooks/                 # Jupyter notebooks (experiments)
│   ├── 01_data_exploration.ipynb
│   ├── 02_classifier_training.ipynb
│   ├── 03_gan_training.ipynb
│   ├── 04_gan_visualization.ipynb
│   └── 05_testing.ipynb
│
├── data/                      # Dataset storage
│   ├── raw/
│   └── processed/
│
├── models/                    # Model architectures & training
│   ├── classifier/
│   ├── gan/
│   └── utils.py
│
├── services/                  # Inference logic
│   ├── predict.py
│   └── generate.py
│
├── api/                       # FastAPI backend
│   ├── main.py
│   ├── routes/
│   └── schemas.py
│
├── frontend/                  # UI (Canvas)
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── utils/                     # Helper functions
│   ├── preprocessing.py
│   └── visualization.py
│
├── experiments/               # Outputs & logs
│   ├── gan_samples/
│   └── plots/
│
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/digit-gan-app.git
cd digit-gan-app
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📓 Running Jupyter Notebooks

```bash
jupyter lab
```

Use notebooks for:

* Data exploration
* Training models
* Visualizing GAN outputs

---

## 🏋️ Training Models

### Train CNN Classifier

```bash
python models/classifier/train.py
```

### Train GAN

```bash
python models/gan/train.py
```

---

## 🌐 Running the Application

### Start Backend

```bash
uvicorn api.main:app --reload
```

### Open Frontend

* Open `frontend/index.html` in your browser

---

## 🎮 How to Use

1. Draw a digit on the canvas
2. Click **Predict**
3. View predicted digit
4. Optionally generate new digits using GAN

---

## 📊 Technologies Used

* Python
* PyTorch
* FastAPI
* Jupyter Notebook / JupyterLab
* OpenCV & Pillow
* HTML, CSS, JavaScript

---

## 🧪 Future Improvements

* ✨ Conditional GAN (control generated digits)
* 📱 Mobile-friendly UI
* ☁️ Deploy on cloud (AWS / GCP)
* 📈 Improve GAN stability (WGAN, DCGAN)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Commit changes
4. Open a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 💡 Acknowledgements

* MNIST Dataset
* PyTorch community
* Open-source ML contributors

---

## 📬 Contact

For questions or collaboration:

* GitHub: https://github.com/your-username

---

⭐ If you found this project useful, consider giving it a star!

