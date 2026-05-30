import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# --- 1. CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(page_title="Hệ Thống Cảnh Báo Cháy", page_icon="🔥", layout="centered")

st.title("🔥 Hệ Thống Phát Hiện Cháy Thời Gian Thực")
st.write("Ứng dụng sử dụng mô hình Deep Learning CNN để nhận diện Lửa/Khói.")
st.markdown("---")

# --- 2. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN ---
@st.cache_resource # Dùng cache để chỉ tải mô hình 1 lần duy nhất, tránh giật lag web
def load_my_model():
    # Sử dụng compile=False nếu lúc load gặp lỗi cấu hình loss custom cũ
    return tf.keras.models.load_model('fire_model.h5', compile=False)

try:
    model = load_my_model()
    st.sidebar.success("✅ Tải mô hình thành công!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải mô hình. Đảm bảo file 'fire_model.h5' nằm cùng thư mục. Chi tiết: {e}")

# Cấu hình sidebar cho người dùng chỉnh ngưỡng nhạy bén (Recall)
threshold = st.sidebar.slider("Ngưỡng cảnh báo cháy (Threshold)", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
st.sidebar.info(f"Ngưỡng hiện tại: {threshold}. Hạ thấp xuống nếu muốn mô hình nhạy bén hơn với lửa (Tăng Recall).")

# --- 3. HÀM TIỀN XỬ LÝ VÀ DỰ ĐOÁN ---
def predict_image(image):
    # Resize về kích thước lúc train (128x128)
    img = image.resize((128, 128))
    img_array = np.array(img)
    
    # Đảm bảo ảnh có đủ 3 kênh màu (RGB)
    if img_array.shape[-1] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Chuẩn hóa giống lúc train
    
    # Dự đoán
    pred = model.predict(img_array, verbose=0)[0]
    
    # LƯU Ý: Đoạn này cấu hình theo index lớp của bạn. 
    # Giả sử index 0 là 'fire' (Đầu ra Softmax 2 nút)
    fire_prob = pred[0] 
    return fire_prob

# --- 4. GIAO DIỆN CHỨC NĂNG ---
tab1, tab2 = st.tabs(["📸 Kiểm tra qua Ảnh tải lên", "🎥 Nhận diện trực tiếp qua Webcam"])

# --- TAB 1: TẢI ẢNH LÊN ---
with tab1:
    st.header("Tải ảnh từ máy tính")
    uploaded_file = st.file_uploader("Chọn một bức ảnh cần kiểm tra...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        with st.spinner('Đang phân tích ảnh...'):
            fire_probability = predict_image(image)
            
        st.subheader("📊 Kết quả phân tích:")
        st.write(f"Xác suất có cháy: **{fire_probability * 100:.2f}%**")
        
        # Áp dụng ngưỡng linh hoạt được chọn từ Sidebar
        if fire_probability > threshold:
            st.error("🚨 CANH BÁO: PHÁT HIỆN CÓ CHÁY TRONG ẢNH!")
        else:
            st.success("🟢 AN TOÀN: Không phát hiện dấu hiệu cháy.")

# --- TAB 2: WEBCAM REAL-TIME ---
# --- TAB 2: WEBCAM REAL-TIME ---
with tab2:
    st.header("Sử dụng Webcam máy tính")
    
    # 🌟 THÊM: Cho phép người dùng chọn Index của Camera (0 là camera mặc định, 1, 2 là camera rời)
    camera_choice = st.selectbox(
        "Chọn cổng Camera thiết bị:",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Camera Cổng {x} " + ("(Mặc định)" if x == 0 else "(Camera rời/Phụ)")
    )
    
    run_webcam = st.checkbox("Bật/Tắt Webcam")
    FRAME_WINDOW = st.image([]) # Khung trống để hiển thị luồng video liên tục
    
    if run_webcam:
        # 🌟 THÊM: Truyền index camera được chọn từ dropdown vào OpenCV
        cap = cv2.VideoCapture(camera_choice) 
        
        # Kiểm tra nhanh xem cổng camera được chọn có mở được không
        if not cap.isOpened():
            st.error(f"❌ Không thể mở được Camera tại cổng {camera_choice}. Vui lòng kiểm tra lại kết nối thiết bị hoặc chọn cổng khác!")
            run_webcam = False
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Mất tín hiệu kết nối hoặc luồng dữ liệu hình ảnh bị ngắt.")
                break
                
            # OpenCV dùng BGR, Streamlit dùng RGB nên cần đổi hệ màu để hiển thị đúng
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Dự đoán trực tiếp khung hình hiện tại
            fire_probability = predict_image(pil_img)
            
            # Vẽ thông tin cảnh báo đè lên frame ảnh để hiển thị ra web
            if fire_probability > threshold:
                cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], frame_rgb.shape[0]), (255, 0, 0), 15)
                text = f"CANH BAO: CO CHAY! ({fire_probability*100:.1f}%)"
                cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                text = f"An Toan ({fire_probability*100:.1f}%)"
                cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Cập nhật khung hình liên tục lên giao diện web
            FRAME_WINDOW.image(frame_rgb, use_container_width=True)
            
        cap.release()
    else:
        st.write("Đã tắt Webcam. Hãy tích vào ô phía trên để kích hoạt.")