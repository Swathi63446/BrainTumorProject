# 🧠 Brain Tumor Classification using Deep Learning

A web-based application that uses Convolutional Neural Networks (CNN) to classify brain MRI scans and detect the presence of brain tumors. The application provides a user-friendly interface for uploading medical images and receiving instant predictions.

## 🚀 Features

- **Real-time Brain Tumor Detection**: Upload MRI scans and get instant predictions
- **Deep Learning Model**: CNN-based architecture trained on 3,000+ medical images
- **Web Interface**: Clean, responsive UI built with Flask and Bootstrap
- **Image Preprocessing**: Automatic image resizing and normalization
- **Binary Classification**: Distinguishes between "No Brain Tumor" and "Brain Tumor Present"

## 🛠️ Technologies Used

### Backend
- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **Flask** - Web framework
- **OpenCV** - Image processing
- **PIL (Pillow)** - Image manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Data preprocessing

### Frontend
- **HTML5/CSS3**
- **Bootstrap 4** - Responsive design
- **JavaScript/jQuery** - Interactive functionality

### Machine Learning
- **Convolutional Neural Network (CNN)**
- **Categorical Cross-Entropy Loss**
- **Adam Optimizer**
- **Data Augmentation & Normalization**

## 📁 Project Structure

```
BrainTumor Classification DL/
├── app.py                          # Main Flask application
├── mainTrain.py                    # Model training script
├── mainTest.py                     # Model testing script
├── requirements.txt                # Python dependencies
├── BrainTumor10EpochsCategorical.h5 # Pre-trained model
├── datasets/                       # Training data
│   ├── no/                        # No tumor images (1500 files)
│   └── yes/                       # Tumor images (1500 files)
├── templates/                      # HTML templates
│   ├── index.html                 # Main page
│   └── import.html                # Base template
├── static/                        # Static assets
│   ├── css/                       # Stylesheets
│   └── js/                        # JavaScript files
└── uploads/                       # User uploaded images
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "BrainTumor Classification DL"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## 🧠 Model Architecture

The CNN model consists of:

1. **Input Layer**: 64x64x3 RGB images
2. **Convolutional Layer 1**: 32 filters (3x3), ReLU activation
3. **Max Pooling Layer 1**: 2x2 pool size
4. **Convolutional Layer 2**: 32 filters (3x3), ReLU activation
5. **Max Pooling Layer 2**: 2x2 pool size
6. **Convolutional Layer 3**: 64 filters (3x3), ReLU activation
7. **Max Pooling Layer 3**: 2x2 pool size
8. **Flatten Layer**: Converts 2D to 1D
9. **Dense Layer**: 64 neurons, ReLU activation
10. **Dropout Layer**: 0.5 dropout rate
11. **Output Layer**: 2 neurons, Softmax activation

## 📊 Dataset Information

- **Total Images**: 3,000 MRI scans
- **Classes**: 2 (No Tumor: 1,500 images, Tumor Present: 1,500 images)
- **Image Format**: JPG
- **Image Size**: Resized to 64x64 pixels
- **Train-Test Split**: 80% training, 20% testing

## 🔧 Usage

1. **Upload Image**: Click "Choose File" and select a brain MRI scan
2. **Preview**: The uploaded image will be displayed
3. **Predict**: Click "Predict!" button to analyze the image
4. **Result**: The application will display whether a brain tumor is detected or not

## 🎯 Model Performance

- **Architecture**: 3-layer CNN with dropout regularization
- **Training**: 10 epochs with batch size 16
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Validation**: 20% holdout validation set

## 🔬 Technical Details

### Image Preprocessing Pipeline
1. Load image using OpenCV
2. Convert to RGB format
3. Resize to 64x64 pixels
4. Normalize pixel values to [0,1] range
5. Expand dimensions for batch processing

### API Endpoints
- `GET /` - Main application page
- `POST /predict` - Image upload and prediction endpoint

## 🚨 Important Notes

- **Medical Disclaimer**: This application is for educational purposes only and should not be used for actual medical diagnosis
- **Image Requirements**: Supports PNG, JPG, and JPEG formats
- **Model Limitations**: Trained on a specific dataset and may not generalize to all brain MRI variations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

[Your Name] - [Your Email]

## 🙏 Acknowledgments

- Dataset source and contributors
- TensorFlow/Keras community
- Flask documentation
- Medical imaging research community

---

**⚠️ Medical Disclaimer**: This tool is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.
