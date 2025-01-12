import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu từ file
def load_data():
    ly_texts = open("./ly.txt", "r", encoding="utf-8").readlines()
    hoa_texts = open("./hoa.txt", "r", encoding="utf-8").readlines()

    # Tạo nhãn cho các văn bản (0 cho lý, 1 cho hóa)
    ly_labels = [0] * len(ly_texts)
    hoa_labels = [1] * len(hoa_texts)

    # Kết hợp dữ liệu và nhãn lại với nhau
    texts = ly_texts + hoa_texts
    labels = ly_labels + hoa_labels

    return texts, labels

# Tiền xử lý và biến đổi dữ liệu văn bản thành các đặc trưng
def preprocess_data(texts):
    vectorizer = CountVectorizer(stop_words=None)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Hàm phân loại văn bản nhập từ bàn phím
def classify_text(input_text, model, vectorizer):
    # Chuyển văn bản nhập từ bàn phím thành vector đặc trưng
    input_vector = vectorizer.transform([input_text])
    # Dự đoán nhãn của văn bản
    prediction = model.predict(input_vector)
    # Phân loại kết quả
    if prediction == 0:
        return "Lý"
    else:
        return "Hóa"

# Main function
def main():
    # Tải dữ liệu
    texts, labels = load_data()

    # Tiền xử lý dữ liệu
    X, vectorizer = preprocess_data(texts)
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Xây dựng mô hình Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá mô hình
    y_pred = model.predict(X_test)

    # Hiển thị kết quả đánh giá
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Phần nhập văn bản từ bàn phím và phân loại
    input_text = input("Nhập văn bản cần phân loại: ")
    classification_result = classify_text(input_text, model, vectorizer)
    print(f"Văn bản này thuộc chủ đề: {classification_result}")

if __name__ == "__main__":
    main()
