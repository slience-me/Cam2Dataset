# 测试opencv是否正常
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)  # 默认摄像头 (如果有多个摄像头，索引可以修改为其他数字)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取每一帧图像
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取图像")
        break
    
    # 显示图像
    cv2.imshow('Camera Feed', frame)
    
    # 按键 'q' 退出
    key = cv2.waitKey(1)  # 每一帧等待1毫秒
    if key == ord('q'):  # 按 'q' 键退出
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()