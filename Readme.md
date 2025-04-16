Hệ Thống Nhận Diện Ngủ Gật Cho Người Lái Xe
Hệ thống sử dụng mô hình Deep Learning (CNN) để phân loại trạng thái mắt của người lái xe, kết hợp với MediaPipe để nhận diện khuôn mặt và xác định vị trí mắt. Khi phát hiện tài xế nhắm mắt quá lâu (ngủ gật), hệ thống sẽ phát cảnh báo.
Cấu Trúc CNN
3 lớp Conv2D (32 filter) + MaxPooling
2 lớp Conv2D (32 filter) + MaxPooling
3 lớp Fully Connected với Dropout (256, 128, 64 units)
Lớp đầu ra sigmoid với 1 neuron (phân loại nhị phân)
![alt text](image.png)

Cài đặt
opencv-python
numpy
tensorflow
mediapipe
playsound

Đảm bảo đã cài đặt các thư viện cần thiết
Huấn luyện mô hình (nếu chưa có): python train_model.py
Chạy chương trình chính: python main.py
Nhấn phím ESC để thoát

Hình anh ứng dụng
Bình thường
![alt text](<Screenshot 2025-04-16 at 21.29.24.png>)
Nháy mắt
![alt text](<Screenshot 2025-04-16 at 21.30.44.png>)
Cảnh Báo
Chưa bắt được khoảnh khắc