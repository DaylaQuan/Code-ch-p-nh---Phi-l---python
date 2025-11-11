import cv2
import os
import time
import mediapipe as mp
import pickle
import numpy as np

# --- Cấu hình ---
# Số lượng ảnh cần chụp
TONG_SO_ANH = 200

# Thư mục để lưu ảnh (sẽ tự động được tạo)
THU_MUC_LUU = "G:/anh"

# Thư mục để lưu dữ liệu landmarks (pickel)
THU_MUC_DATA = "G:/anh/data"

# Khoảng thời gian giữa các lần chụp (giây)
KHOANG_CACH_CHUP = 1.5

# Cài đặt độ sáng và contrast
DO_SANG = 30  # Giá trị từ -100 đến 100 (dương = sáng hơn)
DO_CONTRAST = -20  # Giá trị từ -100 đến 100 (âm = giảm contrast)

# -----------------

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Điều chỉnh độ sáng và contrast của ảnh
    brightness: -100 đến 100
    contrast: -100 đến 100
    """
    # Chuyển đổi brightness và contrast thành alpha và beta
    # Alpha (contrast): 1.0 = không đổi, >1.0 = tăng contrast, <1.0 = giảm contrast
    # Beta (brightness): 0 = không đổi, dương = sáng hơn, âm = tối hơn
    
    alpha = (contrast + 100) / 100.0
    beta = brightness
    
    # Áp dụng công thức: new_image = alpha * image + beta
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return adjusted

def start_capture():
    # 1. Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(THU_MUC_LUU):
        os.makedirs(THU_MUC_LUU)
        print(f"Đã tạo thư mục: {THU_MUC_LUU}")
    
    if not os.path.exists(THU_MUC_DATA):
        os.makedirs(THU_MUC_DATA)
        print(f"Đã tạo thư mục: {THU_MUC_DATA}")

    # 2. Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3. Khởi động webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam!")
        return

    print("Đã mở webcam. Đưa tay vào camera để bắt đầu chụp tự động!")
    print(f"Cài đặt: Độ sáng +{DO_SANG}, Contrast {DO_CONTRAST}")
    print("Nhấn 'q' để thoát bất cứ lúc nào.")

    so_anh_da_chup = 0
    thoi_gian_chup_cuoi = 0
    tat_ca_du_lieu = []  # Lưu tất cả dữ liệu landmarks

    while True:
        # 4. Đọc khung hình
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình.")
            break

        # Lật ngược hình ảnh
        frame = cv2.flip(frame, 1)
        
        # Điều chỉnh độ sáng và contrast
        frame = adjust_brightness_contrast(frame, DO_SANG, DO_CONTRAST)
        
        # Chuyển BGR sang RGB cho MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý nhận diện tay
        results = hands.process(rgb_frame)
        
        # Sao chép khung hình để vẽ
        frame_hien_thi = frame.copy()
        
        # 5. Vẽ landmarks nếu phát hiện tay
        hand_detected = False
        landmarks_data = []
        
        if results.multi_hand_landmarks:
            hand_detected = True
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Vẽ landmarks lên ảnh
                mp_drawing.draw_landmarks(
                    frame_hien_thi,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Trích xuất tọa độ các điểm landmarks
                hand_data = {
                    'hand_index': hand_idx,
                    'handedness': results.multi_handedness[hand_idx].classification[0].label,
                    'landmarks': []
                }
                
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    hand_data['landmarks'].append({
                        'id': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                landmarks_data.append(hand_data)
        
        # 6. Tự động chụp ảnh khi phát hiện tay
        thoi_gian_hien_tai = time.time()
        
        if hand_detected and so_anh_da_chup < TONG_SO_ANH:
            if thoi_gian_hien_tai - thoi_gian_chup_cuoi >= KHOANG_CACH_CHUP:
                # Tạo tên file
                ten_file = f"{so_anh_da_chup + 1:05d}.jpg"
                duong_dan_luu = os.path.join(THU_MUC_LUU, ten_file)
                
                # Lưu ảnh gốc đã điều chỉnh (không có landmarks vẽ)
                cv2.imwrite(duong_dan_luu, frame)
                
                # Lưu dữ liệu landmarks
                du_lieu_anh = {
                    'image_name': ten_file,
                    'image_index': so_anh_da_chup + 1,
                    'timestamp': thoi_gian_hien_tai,
                    'hands': landmarks_data
                }
                tat_ca_du_lieu.append(du_lieu_anh)
                
                print(f"Đã lưu: {duong_dan_luu} - Phát hiện {len(landmarks_data)} bàn tay")
                
                so_anh_da_chup += 1
                thoi_gian_chup_cuoi = thoi_gian_hien_tai
        
        # 7. Hiển thị thông tin trên màn hình
        if so_anh_da_chup < TONG_SO_ANH:
            if hand_detected:
                cv2.putText(frame_hien_thi, f"DANG CHUP: {so_anh_da_chup} / {TONG_SO_ANH}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_hien_thi, f"Phat hien: {len(landmarks_data)} ban tay", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame_hien_thi, "Dua tay vao de bat dau chup!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame_hien_thi, f"Da chup: {so_anh_da_chup} / {TONG_SO_ANH}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Đã chụp đủ
            cv2.putText(frame_hien_thi, "HOAN TAT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_hien_thi, f"Da chup du {TONG_SO_ANH} anh", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Lưu tất cả dữ liệu vào file pickle
            ten_file_pickle = os.path.join(THU_MUC_DATA, "hand_landmarks.pkl")
            with open(ten_file_pickle, 'wb') as f:
                pickle.dump(tat_ca_du_lieu, f)
            print(f"\nĐã lưu dữ liệu landmarks vào: {ten_file_pickle}")
            print(f"Tổng số ảnh có dữ liệu: {len(tat_ca_du_lieu)}")
            
            # Chờ 2 giây rồi thoát
            cv2.imshow("Webcam - Nhan Dien Tay Tu Dong", frame_hien_thi)
            cv2.waitKey(2000)
            break
        
        # 8. Hiển thị webcam
        cv2.imshow("Webcam - Nhan Dien Tay Tu Dong", frame_hien_thi)
        
        # 9. Chờ phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Đã nhấn 'q', đang thoát...")
            # Lưu dữ liệu hiện có trước khi thoát
            if tat_ca_du_lieu:
                ten_file_pickle = os.path.join(THU_MUC_DATA, "hand_landmarks_partial.pkl")
                with open(ten_file_pickle, 'wb') as f:
                    pickle.dump(tat_ca_du_lieu, f)
                print(f"Đã lưu dữ liệu một phần vào: {ten_file_pickle}")
            break
    
    # 10. Dọn dẹp
    print("Đang đóng webcam...")
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_capture()
