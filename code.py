import cv2
import os
import time

# --- Cấu hình ---
# Số lượng ảnh cần chụp
TONG_SO_ANH = 10

# Thư mục để lưu ảnh (sẽ tự động được tạo)
THU_MUC_LUU = "G:/anh"

# Số giây đếm ngược trước khi bắt đầu
DEM_NGUOC = 5

# -----------------

def start_capture():
    # 1. Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(THU_MUC_LUU):
        os.makedirs(THU_MUC_LUU)
        print(f"Đã tạo thư mục: {THU_MUC_LUU}")

    # 2. Khởi động webcam (0 là webcam mặc định)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam!")
        return

    print("Đã mở webcam. Hướng camera và chuẩn bị...")
    print("Nhấn 's' trên cửa sổ webcam để bắt đầu chụp.")
    print("Nhấn 'q' để thoát bất cứ lúc nào.")

    bat_dau_chup = False
    so_anh_da_chup = 0

    while True:
        # 3. Đọc từng khung hình
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình.")
            break

        # Sao chép khung hình để vẽ lên mà không ảnh hưởng ảnh gốc
        frame_hien_thi = frame.copy()

        if not bat_dau_chup:
            # 3.1. Giai đoạn chờ
            cv2.putText(frame_hien_thi, "Nhan 's' de bat dau", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_hien_thi, "Nhan 'q' de thoat", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 3.2. Giai đoạn đang chụp
            if so_anh_da_chup < TONG_SO_ANH:
                # Tạo tên file (ví dụ: 00001.jpg, 00002.jpg, ...)
                # Số 05d nghĩa là đệm 5 chữ số 0
                ten_file = f"{so_anh_da_chup + 1:05d}.jpg"
                duong_dan_luu = os.path.join(THU_MUC_LUU, ten_file)

                # Lưu ảnh!
                cv2.imwrite(duong_dan_luu, frame)
                
                print(f"Đã lưu: {duong_dan_luu}")

                # Hiển thị thông báo lên màn hình
                cv2.putText(frame_hien_thi, f"Da luu: {so_anh_da_chup + 1} / {TONG_SO_ANH}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                so_anh_da_chup += 1
            else:
                # Đã chụp đủ
                print(f"Hoàn tất! Đã chụp đủ {TONG_SO_ANH} tấm ảnh.")
                break

        # 4. Hiển thị webcam
        cv2.imshow("Webcam - Chup Anh Tu Dong", frame_hien_thi)

        # 5. Chờ phím nhấn
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Đã nhấn 'q', đang thoát...")
            break
        
        if key == ord('s') and not bat_dau_chup:
            print("Bắt đầu đếm ngược...")
            # Giai đoạn đếm ngược
            for i in range(DEM_NGUOC, 0, -1):
                ret_countdown, frame_countdown = cap.read()
                if not ret_countdown:
                    break
                
                # Vẽ số đếm ngược
                text = str(i)
                font_scale = 3
                thickness = 4
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                x = (frame_countdown.shape[1] - text_width) // 2
                y = (frame_countdown.shape[0] + text_height) // 2

                cv2.putText(frame_countdown, text, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                cv2.imshow("Webcam - Chup Anh Tu Dong", frame_countdown)
                cv2.waitKey(1000) # Chờ 1 giây
            
            print("Bắt đầu chụp!")
            bat_dau_chup = True

    # 6. Dọn dẹp
    print("Đang đóng webcam...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_capture()