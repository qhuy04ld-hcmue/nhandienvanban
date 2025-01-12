import os
import tkinter as tk
from tkinter import filedialog
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

# Hàm phân loại văn bản nhập từ file
def classify_text(input_text, model, vectorizer):
    # Chuyển văn bản nhập từ file thành vector đặc trưng
    input_vector = vectorizer.transform([input_text])
    # Dự đoán nhãn của văn bản
    prediction = model.predict(input_vector)
    # Phân loại kết quả
    if prediction == 0:
        return "Lý"
    else:
        return "Hóa"

# Hàm mở cửa sổ chọn file và đọc nội dung từ file
def choose_file_and_classify(model, vectorizer):
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của tkinter
    file_path = filedialog.askopenfilename(title="Chọn file văn bản", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))

    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            input_text = file.read()

        # Phân loại văn bản từ file
        classification_result = classify_text(input_text, model, vectorizer)
        print(f"Văn bản trong file này thuộc chủ đề: {classification_result}")
    else:
        print("Bạn chưa chọn file!")

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

    # Phần chọn file và phân loại
    choose_file_and_classify(model, vectorizer)

if __name__ == "__main__":
    main()
