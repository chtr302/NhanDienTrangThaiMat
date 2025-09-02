# 🎯 Hệ Thống Nhận Diện Ngủ Gật

## ✅ Status: All issues FIXED + UI improved!

> 📖 **READ FIRST:** [`START_HERE.txt`](START_HERE.txt)

## ⚡ Quick Start:
```bash
# 1. fix_venv_issue.bat (setup Python 3.10)
# 2. fix_numpy_issue.bat (fix conflicts)  
# 3. activate_py310.bat → python test_ai_modules.py
# 4. python main_app_safe.py
```

## 📂 Cấu trúc Files:

| File | Mô tả | Khi nào dùng |
|------|-------|--------------|
| `START_HERE.txt` | **ĐỌC ĐẦU TIÊN** - Hướng dẫn đầy đủ | Setup |
| `fix_venv_issue.bat` | **Setup** Python 3.10 environment | Setup lần đầu |
| `fix_numpy_issue.bat` | **Fix** Keras & NumPy conflicts | Khi lỗi import |
| `activate_py310.bat` | **Activate** environment | Trước khi chạy |

| `test_ai_modules.py` | **Test riêng** AI modules | Khi AI lỗi |
| `main_app_safe.py` | **App chính** | Chạy hàng ngày |
| `requirements_py310.txt` | Dependencies (no standalone Keras) | Reference |

## 🎨 Tính năng:

### Panel Bên Trái (Settings):
- ✅ Cài đặt âm thanh (file, thời gian)
- ✅ Nhận diện mắt (ngưỡng thời gian) 
- ✅ Nhận diện ngáp (bật/tắt, số lần tối đa)
- ✅ Điều khiển (Start/Stop)

### Panel Bên Phải (Cameras):
- ✅ Camera trên: **Vùng mắt** (640x160) với trạng thái real-time
- ✅ Camera dưới: Nhận diện ngáp (placeholder khi tắt)

## 🛠️ Cài đặt Step-by-step:

### **Option 1: Automatic (Khuyến nghị)**
```bash
# 1. Cài Python 3.10 từ python.org
# 2. Double-click: setup_venv.bat
# 3. Double-click: run_with_py310.bat
```

### **Option 2: Manual**
```bash
# 1. Tạo venv với Python 3.10
py -3.10 -m venv venv_py310

# 2. Kích hoạt
venv_py310\Scripts\activate.bat

# 3. Cài dependencies
pip install -r requirements_py310.txt

# 4. Chạy app
python main_app_safe.py
```

## 🔧 Troubleshooting:

**❌ Lỗi chính: Python 3.12 incompatible**
- **Giải pháp:** Setup Python 3.10 theo hướng dẫn trên

**❌ 'py -3.10' not found**
- **Giải pháp:** Cài Python 3.10 từ python.org, tick "Add to PATH"

**Nếu không có camera:**
- App vẫn chạy được, chỉ hiện thông báo lỗi camera

**Nếu muốn train model ngáp:**
```bash
python train_yawn_model.py
```

## 📞 Support:

- Safe mode tự động handle lỗi AI
- Có fallback cho mọi trường hợp
- UI luôn hoạt động dù AI lỗi

---
*Made with ❤️ using PyQt6 + OpenCV + TensorFlow*
