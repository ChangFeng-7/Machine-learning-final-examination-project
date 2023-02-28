import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from PIL import Image, ImageTk
import  cv2
from tkinter import filedialog, dialog
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms




def image_resize(img, screen_width=300, screen_height=200):
    image = img

    raw_width, raw_height = image.size[0], image.size[1]
    max_width, max_height = raw_width, screen_height
    min_width = max(raw_width, max_width)
    # 按照比例缩放
    min_height = int(raw_height * min_width / raw_width)
    # 第1次快速调整
    while min_height > screen_height:
        min_height = int(min_height * .9533)
    # 第2次精确微调
    while min_height < screen_height:
        min_height += 1
    # 按照比例缩放
    min_width = int(raw_width * min_height / raw_height)
    # 适应性调整
    while min_width > screen_width:
        min_width -= 1
    # 按照比例缩放
    min_height = int(raw_height * min_width / raw_width)
    return image.resize((min_width, min_height))


def open_file_output():
    '''
    打开文件
    :return:local_
    '''
    global file_path
    global file_text
    global photo
    global img
    file_path = filedialog.askopenfilename(title=u'选择水果图片')
    print('打开文件：', file_path)
    if file_path is not None:
        file_text = "文件路径为：" + file_path

    img = Image.open(file_path)  # 打开图片
    img = image_resize(img)
    photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开



    imglabel = tkinter.Label(window, bd=10, image=photo)
    imglabel.place(relx=0.1, rely=0.1)



def run():
    path_temp = file_path
    # path_temp = self.datapath + '\\' + self.img_path[index]
    data = Image.open(path_temp).convert('RGB')
    data = transform(data)
    data = data.unsqueeze(0)
    output = model(data)
    _, preds = torch.max(output, 1)

    if preds == 0 :
        t = "该窗帘为没缺陷"
    else:
        t = "该窗帘为有缺陷"

    #
    global output_text
    output_text= t

    text = tkinter.Label(window, bd=10, font=40, fg='red', bg='white', text=t)

    text.place(relx=0.6, rely=0.3)  # 相对位置，放置文本

def load_model():
    text1 = tkinter.Label(window, bd=10, font=40, text = "模型加载成功")

    text1.place(relx=0.2, rely=0.8)


def save_img():
    save_output_path = r'.\output\%s' % (file_text.split('/')[-1])  # 未过滤目标
    plt.figure()
    plt.subplot(1, 2, 1)

    img2 = Image.open(file_path).convert("RGB") # plt可以处理PIL下的Image文件，变色问题记得加.convert

    plt.axis('off')
    plt.imshow(img2)
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.45)

    plt.subplot(1, 2, 2)
    plt.rcParams['font.sans-serif'] = ['simhei']  # 设置黑体
    plt.rcParams['axes.unicode_minus'] = False  # 关闭

    plt.axis('off')
    plt.title(output_text, fontsize=20,x=0.5,y=0.5)
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.45)


    plt.savefig(save_output_path, dpi=1200, bbox_inches='tight')


    # plt.show()
    plt.close()

    text2 = tkinter.Label(window, bd=10, font=40, text="保存成功")

    text2.place(relx=0.63, rely=0.8)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    model_path = ".\save_model\\0025_acc_0.8667.pth"
    # model = AlexNet().to(device)
    model = torch.load(model_path, map_location='cpu')

    model.eval()

    window = tkinter.Tk()
    window.title('窗帘缺陷检测')
    window.geometry('600x500')

    button1 = tkinter.Button(window, text='选择窗帘图片', command=open_file_output, width=10, height=2)  # 加括号会自动执行（！！）
    button5 = tkinter.Button(window, text='退出',bg = "red",fg = "white", command=lambda: window.destroy(), width=10, height=2)
    button3 = tkinter.Button(window, text='处理', command=run, width=10, height=2)  # 加括号会自动执行（！！）
    button2 = tkinter.Button(window, text='加载模型', command=load_model, width=10, height=2)
    button4 = tkinter.Button(window, text='结果保存', command=save_img, width=10, height=2)

    button1.place(relx=0.17, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button2.place(relx=0.37, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button3.place(relx=0.57, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button4.place(relx=0.77, rely=0.8, anchor='se')  # 相对位置，放置按钮
    button5.place(relx=0.97, rely=0.8, anchor='se')  # 相对位置，放置按钮




window.mainloop()




