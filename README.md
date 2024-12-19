# Part 1 - Artificial Neural Networks 

## ANN Intuition
***1. What You'll Need for ANN***
[Link tài liệu](https://www.superdatascience.com/blogs/the-ultimate-guide-to-artificial-neural-networks-ann)

***2. How Neural Networks Learn: Gradient Descent and Backpropagation Explained***
***1. Nơ-ron***
- Tìm hiểu cơ bản về nơ-ron và cách hoạt động của não người.  
- Khám phá nơ-ron, khối xây dựng chính của mạng nơ-ron.  

***2. Hàm Kích Hoạt***  
- Khám phá các hàm kích hoạt phổ biến trong mạng nơ-ron.  
- Tìm hiểu cách sử dụng và vị trí áp dụng các hàm kích hoạt.  

***3. Cách Mạng Nơ-ron Hoạt Động***  
- Hiểu cách mạng nơ-ron vận hành qua ví dụ dự đoán giá bất động sản.  
- Nắm rõ mục tiêu mà mạng nơ-ron hướng tới.  

***4. Cách Mạng Nơ-ron Học***  
- Tìm hiểu quá trình học tập và cải thiện của mạng nơ-ron.  

***5. Gradient Descent***  
- Hiểu thuật toán Gradient Descent.  
- Lý giải vì sao nó hiệu quả hơn phương pháp brute force.  

***6. Stochastic Gradient Descent (SGD)***  
- Tìm hiểu phiên bản cải tiến của Gradient Descent.  
- Hiểu cách SGD cải thiện hiệu suất học tập.  

***7. Backpropagation***  
- Tóm tắt các khái niệm quan trọng về backpropagation.  
- Trình tự thực hiện mạng nơ-ron nhân tạo từng bước.

## Tổng Quan Về Nơ-ron – Bài Học Cơ Bản Về Mạng Nơ-ron Nhân Tạo  

![image](https://github.com/user-attachments/assets/c8cc3dfe-ea5b-481a-ba33-bd5dff988bee)

***1. Nơ-ron Sinh Học***  
- Nơ-ron là đơn vị cơ bản của bộ não, với cấu trúc gồm thân, các nhánh dendrite (nhận tín hiệu), và đuôi axon (truyền tín hiệu).  
- Santiago Ramón y Cajal lần đầu tiên phác thảo cấu trúc nơ-ron vào năm 1899, dựa trên quan sát thực tế.  
- Các nơ-ron không hoạt động đơn lẻ; chúng phối hợp qua các kết nối gọi là synapse để truyền tín hiệu.  

***2. Nơ-ron Trong Máy Học*** 
- **Nơ-ron nhân tạo (node)**:  
  - Nhận tín hiệu đầu vào (input), xử lý qua các **trọng số (weights)** và **hàm kích hoạt (activation function)**, sau đó truyền tín hiệu ra đầu ra (output).  
- **Cấu trúc mạng**:  
  - Tầng đầu vào (input layer): Nhận giá trị từ các biến độc lập (independent variables).  
  - Tầng ẩn (hidden layer): Xử lý tín hiệu trung gian.  
  - Tầng đầu ra (output layer): Cung cấp kết quả, có thể là liên tục, nhị phân hoặc phân loại.  

***3. Quá Trình Hoạt Động Của Nơ-ron***  
1. **Nhận tín hiệu đầu vào**:  
   - Các giá trị đầu vào được chuẩn hóa (normalization) hoặc tiêu chuẩn hóa (standardization) để tối ưu hóa quá trình học.  
2. **Tính toán tổng có trọng số**:  
   - Cộng các giá trị đầu vào, mỗi giá trị nhân với trọng số tương ứng.  
3. **Áp dụng hàm kích hoạt**:  
   - Xác định xem tín hiệu có được truyền tiếp hay không, dựa trên giá trị tổng.  
4. **Truyền tín hiệu ra đầu ra**:  
   - Tín hiệu được truyền đến các nơ-ron tiếp theo thông qua các synapse.  

***4. Ý Nghĩa Của Trọng Số***  
- Trọng số quyết định mức độ quan trọng của mỗi tín hiệu đầu vào.  
- Trong quá trình học (gradient descent, backpropagation), các trọng số được điều chỉnh để tối ưu hóa kết quả.  

***5. Ứng Dụng Cụ Thể***  
- Đầu vào có thể là các biến như tuổi, thu nhập, hay phương tiện di chuyển.  
- Đầu ra có thể là giá trị liên tục (ví dụ: giá nhà), giá trị nhị phân (ví dụ: rời khỏi hay ở lại), hoặc phân loại (ví dụ: loại xe).
  
# Tóm tắt bài giảng: Cách hoạt động của mạng nơ-ron

***1. Cấu trúc mạng nơ-ron***

- **Dữ liệu đầu vào**: Bao gồm 4 tham số:
  - Diện tích
  - Số phòng ngủ
  - Khoảng cách đến thành phố
  - Tuổi của bất động sản

- **Lớp đầu ra**: Giá trị được dự đoán (giá của bất động sản).

- **Lớp ẩn**: Cải thiện khả năng mô hình hóa bằng cách kết hợp và trích xuất đặc trưng từ các tham số đầu vào.

---

***2. Cách hoạt động của mạng***

***2.1 Lớp ẩn và các nơ-ron***
- Mỗi nơ-ron trong lớp ẩn tập trung vào một tổ hợp tham số cụ thể, phản ánh các mối quan hệ phức tạp trong dữ liệu.
- **Ví dụ**:
  - **Nơ-ron 1**: Tập trung vào diện tích và khoảng cách đến thành phố, phản ánh mối quan hệ giữa diện tích và giá trị theo khoảng cách.
  - **Nơ-ron 2**: Kết hợp diện tích, số phòng ngủ, và tuổi của bất động sản để xác định nhu cầu của gia đình trẻ muốn nhà mới, rộng rãi.
  - **Nơ-ron 3**: Chỉ chú trọng tuổi của bất động sản, phát hiện sự khác biệt giữa nhà cũ thông thường và nhà cổ có giá trị lịch sử.

***2.2 Chức năng kích hoạt***
- Giúp nơ-ron "kích hoạt" khi phát hiện các mẫu phù hợp trong dữ liệu và đóng góp vào đầu ra.

---

***3. Ý nghĩa của lớp ẩn***

- **Tăng cường độ chính xác**: Phát hiện các mẫu phức tạp và tạo ra các đặc trưng mới từ dữ liệu ban đầu.
- Mạng nơ-ron giống như một cộng đồng kiến:
  - Mỗi nơ-ron góp phần nhỏ.
  - Cùng nhau tạo ra kết quả mạnh mẽ và chính xác hơn.

## Cách Mạng Nơ-ron Học Tập

***1. Giới thiệu***

Có hai cách để lập trình máy tính thực hiện một nhiệm vụ:
1. **Hard-coded**: Viết các quy tắc cụ thể và hướng dẫn rõ ràng cho từng tình huống.
2. **Học máy (Neural networks)**: Xây dựng mạng nơ-ron, cung cấp dữ liệu đầu vào và đầu ra, để mạng tự học.

---

***2. Ví dụ minh họa***

## Phân biệt mèo và chó:
- **Cách 1 (Hard-coded)**: Viết quy tắc chi tiết như hình dạng tai, râu, khuôn mặt, màu sắc.
- **Cách 2 (Neural networks)**: Cung cấp ảnh đã được gán nhãn và để mạng tự học phân biệt.

---

***3. Mạng Perceptron Đơn Giản***

- **Perceptron**: Mạng nơ-ron một lớp (feedforward), do Frank Rosenblatt phát minh năm 1957.
- **Ký hiệu:**
  - \( y \): Giá trị thực (actual value).
  - \( \hat{y} \): Giá trị dự đoán (predicted value).

---

***4. Học qua Backpropagation***

***Quy trình học:***
1. **Dự đoán**: Mạng nơ-ron tính toán \( \hat{y} \) từ dữ liệu đầu vào.
2. **Tính lỗi**: Sử dụng hàm chi phí (cost function), ví dụ:  
   <img width="155" alt="image" src="https://github.com/user-attachments/assets/dd7babfa-ac80-4d0e-9877-f1828de1cb34" />
3. **Cập nhật trọng số**: Dựa trên lỗi, điều chỉnh các trọng số:
<img width="155" alt="image" src="https://github.com/user-attachments/assets/604e5def-cf8a-4ce9-8630-ebfa86d82018" />

***Minh họa đơn giản:***
- Đầu vào: Giờ học, giờ ngủ, điểm kiểm tra giữa kỳ.
- Đầu ra: Điểm thi cuối kỳ.
- Lặp lại quá trình trên với cùng một hàng dữ liệu, cho đến khi lỗi giảm về gần 0.

---

***5. Học trên tập dữ liệu nhiều hàng***

***Một Epoch:***
Khi toàn bộ tập dữ liệu được đưa vào mạng để huấn luyện.

***Quy trình:***
1. Đưa từng hàng dữ liệu vào mạng.
2. Dự đoán \( \hat{y} \), so sánh với \( y \), tính lỗi.
3. Tính tổng lỗi từ tất cả các hàng (hàm chi phí tổng quát).
4. Cập nhật trọng số cho tất cả các hàng dựa trên lỗi tổng.

---

***6. Mục tiêu của mạng***

- Tối ưu hóa trọng số để giảm hàm chi phí.
- Khi hàm chi phí đạt giá trị tối thiểu, trọng số được coi là tốt nhất để áp dụng vào giai đoạn kiểm thử.

---

***7. Tài liệu tham khảo***

Tìm hiểu thêm về các hàm chi phí trên bài viết: _“A list of cost functions used in neural networks alongside applications”_.





