# American Sign Language Detection using OpenCV, Mediapipe, and Scikit-Learn

## 📌 Project Overview
This project is a **real-time American Sign Language (ASL) alphabet detection system** that uses computer vision and machine learning to recognize hand gestures. The system leverages **OpenCV** for webcam input, **Mediapipe** for hand landmark detection, and **Scikit-Learn** to classify ASL alphabet signs.

## 🔑 Features
- Real-time ASL alphabet detection (A, B, L)
- Hand landmark detection using **Mediapipe Hands**
- Machine Learning model using **RandomForestClassifier**
- Bounding box and character display on the detected hand
- Lightweight and optimized for real-time performance

## 🛠️ Tech Stack
- Python
- OpenCV
- Mediapipe
- Scikit-Learn
- Pickle (for model serialization)

## 🎯 Prerequisites
Before running the project, ensure you have the following installed:

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

## 📄 Project Structure
```plaintext
|-- model.p                   # Pre-trained RandomForestClassifier model
|-- inference_classifier.py    # Main ASL Detection Script
|-- README.md                 # Project Documentation
```

## ⚙️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/asl-detector.git
cd asl-detector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Place the `model.p` file in the project directory.

4. Run the detection script:
```bash
python inference_classifier.py
```

5. Press **'q'** to quit the webcam window.

## 🎯 Model Explanation
The machine learning model is built using **RandomForestClassifier** from Scikit-Learn. It takes **42 landmark features** extracted from Mediapipe (21 hand landmarks with (x, y) coordinates).

The model is trained to classify three ASL signs:
- A
- B
- L

## 📌 How It Works
1. The webcam captures live video.
2. Mediapipe detects hand landmarks.
3. Hand landmark coordinates are normalized.
4. The feature array is fed into the **RandomForestClassifier** model.
5. The predicted sign is displayed on the video frame.

## 🔑 Example Output
```
Detected Sign: A
Detected Sign: B
Detected Sign: L
```
🖼️ Sample Photo

Below is a sample screenshot of the ASL detection in action:

![Screenshot 2025-03-08 011620](https://github.com/user-attachments/assets/e8878911-ffbd-4fe7-b018-2fdc5d1723a1)


![Screenshot 2025-03-08 011641](https://github.com/user-attachments/assets/96256dcb-eaee-4f5e-a80c-146b679c4c1d)

![Screenshot 2025-03-08 011724](https://github.com/user-attachments/assets/618ffa53-87b4-4d5f-839e-5da61a9e2817)


## 🚀 Future Improvements
- Add more ASL alphabets
- Improve model accuracy with more training data
- Implement dynamic gesture recognition

## 🤝 Contributing
Pull requests are welcome! Feel free to submit any bug fixes or feature improvements.

## 📄 License
This project is licensed under the MIT License.

