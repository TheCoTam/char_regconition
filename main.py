# import for window
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
# import for model
import cv2
import keras
import pickle
import imutils
import numpy as np
from matplotlib import pyplot as plt


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('700x500')
        self.root.title('Tôi còn không biết đây là cái gì')
        self.mainframe = tk.Frame(self.root, background='white')
        self.mainframe.pack(fill='both', expand=True)

        # Load the original image
        plus_icon = tk.PhotoImage(file="meow-meow.gif")
        # plus_icon = plus_icon.subsample(2)

        # Create a button with the image
        self.button = ttk.Button(self.mainframe, image=plus_icon, command=self.open_image)
        self.button.pack()

        # Create a frame for the label
        label_frame = tk.Frame(self.mainframe)
        label_frame.pack(fill='both', expand=True)

        check_button = tk.Button(label_frame, text="Check!!!", command=self.print_text)
        check_button.grid(row=0, column=0, sticky=tk.W)

        self.label = tk.Label(label_frame, text="")
        self.label.grid(row=3, column=2, sticky=tk.W)

        # Đặt căn giữa cho nhãn
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Đặt kích thước font lớn hơn
        font = ("Arial", 30)
        self.label.config(font=font)

        self.filename = "Thay thế ảnh!!!"  # Thêm thuộc tính filename

        self.root.mainloop()

    def open_image(self):
        # Open file dialog to select an image file
        filetypes = (("Image files", "*.jpg;*.jpeg;*.png;*.gif"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(filetypes=filetypes)
        self.filename = filename

        if filename:
            # Load the selected image
            image = Image.open(filename)

            # Resize the image if it's larger than the maximum size
            max_width = 400
            max_height = 300
            if image.width > max_width or image.height > max_height:
                image.thumbnail((max_width, max_height))

            # Convert the image to a format compatible with tkinter
            image_tk = ImageTk.PhotoImage(image)

            # Update the button's image
            self.button.configure(image=image_tk)
            self.button.image = image_tk

    def print_text(self):
        letter, res_image = get_letters(self.filename)
        word = get_word(letter)
        self.label.configure(text=word)
        plt.imshow(res_image)
        # self.label.configure(text=self.filename)


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # tính toán hình chữ nhật cao quanh đường viền cho mỗi phần tử trong danh sách cnts và lưu va danh sách boundingBoxes
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # sắp xếp danh sách cnts và boundingBoxes dựa trên toạ độ theo chiều ngang
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def get_letters(img):
    letters = []
    # đọc ảnh bằng opencv
    image = cv2.imread(img)
    # chuyển ảnh được đọc vào sang ảnh xám (tác dụng: đơn giản hoá quá trình xử lý, tăng tốc độ xử lý, tập trung vào độ sáng)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # tách biệt vùng chữ và nền bằng cách áp dụng ngưỡng nhị phân đảo lên ảnh xám
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # làm mờ và mở rộng các đối tượng trắng trong ảnh -> dễ xác định hơn
    dilated = cv2.dilate(thresh1, None, iterations=2)

    # tìm các đường viền trong bản sao của ảnh (tìm đường viền bên ngoài)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # chuyển định dạng của biến cnts thành kiểu list
    cnts = imutils.grab_contours(cnts)
    # sắp xếp theo thứ tự
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # lặp qua tập các đường viền để dự đoán ký tự
    for c in cnts:
        # loại bỏ các đường viên có kích thước nhỏ (ở dưới là <= 10)
        if cv2.contourArea(c) > 10:
            # tính toán hình chữ nhật bao quanh đường viền
            (x, y, w, h) = cv2.boundingRect(c)
            # vẽ hình chữ nhật bao quanh đường viền trên ảnh gốc
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cắt phần ký tự xác định được và lưu vào biến roi
        roi = gray[y:y + h, x:x + w]
        # tách biệt phần ký tự khỏi nền bằng cách áp dụng phương pháp ngưỡng hoá
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # thay đổi kích thước ảnh thành 32x32
        thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
        # chuyển đổi kiểu dữ liệu sang foat32 và chuẩn hoá giá trị pixel trong phạm vi 0 -> 1
        thresh = thresh.astype("float32") / 255.0
        # thêm 1 chiều có giá trị là 1 vào cuối
        thresh = np.expand_dims(thresh, axis=-1)
        # thay đổi hình dạng của mảng
        thresh = thresh.reshape(1, 32, 32, 1)
        ypred = model.predict(thresh)
        ypred = LB.inverse_transform(ypred)
        [x] = ypred
        letters.append(x)
    return letters, image


def get_word(letter):
    word = "".join(letter)
    return word


model = keras.models.load_model("char_regconition.keras")
# model = keras.models.load_model("duy.keras")
with open('label_binarizer.pkl', 'rb') as file:
    LB = pickle.load(file)


if __name__ == '__main__':
    app = App()
