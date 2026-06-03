import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Hệ Thống Cảnh Báo Cháy", page_icon="🔥", layout="centered")

st.title("🔥 Hệ Thống Phát Hiện Lửa")
st.write("Ứng dụng sử dụng mô hình Deep Learning CNN để nhận diện Lửa.")
st.markdown("---")

# Cố định ngưỡng cảnh báo mặc định
THRESHOLD = 0.5

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('fire_model.h5', compile=False)

try:
    model = load_my_model()
    st.sidebar.success("✅ Tải mô hình thành công!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải mô hình. Đảm bảo file 'fire_model.h5' nằm cùng thư mục. Chi tiết: {e}")

def predict_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img)

    if img_array.shape[-1] != 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    pred = model.predict(img_array, verbose=0)[0]
    fire_prob = pred[0] 
    return fire_prob

tab1, tab2, tab3 = st.tabs(["📸 Kiểm tra qua Ảnh", "🎥 Nhận diện qua Video tải lên", "💻 Nhận diện trực tiếp qua Webcam"])

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
        
        if fire_probability > THRESHOLD:
            st.error("🚨 CANH BÁO: PHÁT HIỆN CÓ CHÁY TRONG ẢNH!")
        else:
            st.success("🟢 AN TOÀN: Không phát hiện dấu hiệu cháy.")

with tab2:
    st.header("Tải video từ máy tính")
    uploaded_video = st.file_uploader("Chọn một file video cần kiểm tra...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())

        vf_info = cv2.VideoCapture(tfile.name)
        total_frames = int(vf_info.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vf_info.get(cv2.CAP_PROP_FPS)
        vf_info.release()
        
        st.info(f"🎞️ Thông số video: Tổng số {total_frames} frames | Tốc độ: {fps:.2f} FPS")

        start_frame = st.slider(
            "⏩ Kéo thanh trượt để tua Video (Chọn khung hình bắt đầu):", 
            min_value=0, 
            max_value=total_frames - 1, 
            value=0,
            step=1
        )

        col1, col2 = st.columns(2)
        with col1:
            play_video = st.checkbox("Chạy / Tiếp tục phân tích", value=True)
        with col2:
            st.caption(f"Đang xem tại vị trí frame: **{start_frame}** / {total_frames}")

        VIDEO_WINDOW = st.image([])

        vf = cv2.VideoCapture(tfile.name)

        vf.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if play_video:
            current_frame_idx = start_frame
            
            while vf.isOpened() and play_video:
                ret, frame = vf.read()
                if not ret:
                    st.warning("Đã chạy hết video hoặc luồng dữ liệu kết thúc.")
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                fire_probability = predict_image(pil_img)

                if fire_probability > THRESHOLD:
                    cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], frame_rgb.shape[0]), (255, 0, 0), 15)
                    text = f"CANH BAO: CO CHAY! ({fire_probability*100:.1f}%)"
                    cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    text = f"An Toan ({fire_probability*100:.1f}%)"
                    cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                VIDEO_WINDOW.image(frame_rgb, use_container_width=True)
                
                current_frame_idx += 1
                
        else:
            ret, frame = vf.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                fire_probability = predict_image(pil_img)
                
                if fire_probability > THRESHOLD:
                    cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], frame_rgb.shape[0]), (255, 0, 0), 15)
                    text = f"CANH BAO (TAM DUNG): {fire_probability*100:.1f}%"
                    cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    text = f"An Toan (Tam Dung): {fire_probability*100:.1f}%"
                    cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                VIDEO_WINDOW.image(frame_rgb, use_container_width=True)
                
        vf.release()

with tab3:
    st.header("Sử dụng Webcam thiết bị")
    
    camera_choice = st.selectbox(
        "Chọn cổng Camera thiết bị:",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"Camera Cổng {x} " + ("(Mặc định)" if x == 0 else "(Camera rời/Phụ)")
    )
    
    run_webcam = st.checkbox("Bật/Tắt Webcam")
    FRAME_WINDOW = st.image([]) 
    
    if run_webcam:
        cap = cv2.VideoCapture(camera_choice) 
        
        if not cap.isOpened():
            st.error(f"❌ Không thể mở được Camera tại cổng {camera_choice}. Vui lòng kiểm tra lại kết nối thiết bị hoặc chọn cổng khác!")
            run_webcam = False
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Mất tín hiệu kết nối hoặc luồng dữ liệu hình ảnh bị ngắt.")
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            fire_probability = predict_image(pil_img)
            
            if fire_probability > THRESHOLD:
                cv2.rectangle(frame_rgb, (0, 0), (frame_rgb.shape[1], frame_rgb.shape[0]), (255, 0, 0), 15)
                text = f"CANH BAO: CO CHAY! ({fire_probability*100:.1f}%)"
                cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                text = f"An Toan ({fire_probability*100:.1f}%)"
                cv2.putText(frame_rgb, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            FRAME_WINDOW.image(frame_rgb, use_container_width=True)
            
        cap.release()