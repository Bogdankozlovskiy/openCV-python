from tkinter import *
from PIL import ImageTk, Image
import cv2
from tkinter import filedialog, PhotoImage
import numpy as np
from matplotlib.pyplot import imsave


root = Tk()

label_name_file = Label(root)
label_in = Label(root, text='in: 0')
label_out = Label(root, text='out: 0')
frame_count = Label(root, text='frame: ')

frame_count.grid(row=0, column=1)
label_name_file.grid(row=0, column=2)
label_in.grid(row=0, column=3)
label_out.grid(row=0, column=4)
flag = True


def check(x):
    x = [i[0] for i in x]
    return max(x) - min(x) > 5

def centropid(arr, default_centroid):
    d = [i for i in arr if check(i)]
    arr = [i[-1] for i in d]
    first = [i[0] for i in arr]
    second = [i[1] for i in arr]
    if first and second:
        return int(sum(first) // len(first)), int(sum(second) // len(second))
    else:
        return default_centroid
    
def get_param_for_start():
    #start work
    xl, xr, yb, yh = 210, 310, 200, 300
    default_centroid = (xl + xr) // 2, (yb + yh) // 2
    # Количество отслеживаемых кадров
    num_frames_to_track = 5
    # шаг пропуска
    num_frames_jump = 2
    # Инициализация переменных
    tracking_paths = []
    frame_index = 0
    frame_index_write = 0
    in_ = 0
    out_ = 0
    # Определение параметров отслеживания
    tracking_params = dict(winSize  = (11, 11), 
                           maxLevel = 2,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, .03))
    
    return (frame_index_write, num_frames_jump, tracking_paths, xl, xr, yb, yh, default_centroid, 
            num_frames_to_track, frame_index, in_, out_, tracking_params)

def video_stream(cap, lmain, frame_index_write, num_frames_jump, tracking_paths, xl, xr, yb, yh, 
                 default_centroid, num_frames_to_track, frame_index, in_, out_, tracking_params, prev_gray=None):
    _, frame = cap.read()
    # Преобразование в градации серого
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Создание копии кадра
    output_img = frame.copy()
    if len(tracking_paths) > 0:
        # Получение изображений
        prev_img, current_img = prev_gray, frame_gray
        # Организация особых точек
        feature_points_0 = np.float32([tp[-1] for tp in tracking_paths]).reshape(-1, 1, 2)
        # Вычислим оптический поток на
        # основании предыдущих и текущих изображений, используя особые точки и параметры отслеживания.
        feature_points_1, *_= cv2.calcOpticalFlowPyrLK(prev_img, current_img, feature_points_0,None, **tracking_params)
        # Вычисление обратного оптического потока
        feature_points_0_rev, *_= cv2.calcOpticalFlowPyrLK(current_img, prev_img, feature_points_1,None, **tracking_params)
        # Вычисление разности между прямым
        # и обратным оптическими потоками
        diff_feature_points = abs(feature_points_0 - feature_points_0_rev).reshape(-1, 2).max(-1)
        # Извлечение подходящих точек
        good_points = diff_feature_points < 1
        # Инициализация переменной
        new_tracking_paths = []
        # Итерации по всем подходящим особым точкам
        for tp, (x, y), good_points_flag in zip(tracking_paths, feature_points_1.reshape(-1, 2), good_points):
            # Продолжение, если флаг не равен true
            if not good_points_flag:
                continue
            # Присоединение координат Х и У и проверка того,
            # не превышает ли длина списка пороговое значение
            tp.append((x, y))
            if len(tp) > num_frames_to_track:
                del tp[0]
            new_tracking_paths.append(tp)
            # Вычерчивание окружности вокруг особых точек
            cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)
        # Обновление путей отслеживания
        tracking_paths = new_tracking_paths
        #фиксируем выход из облости############################################################################
        if centropid(tracking_paths, default_centroid)[0] < xl - 50:
            if frame_index - frame_index_write > 30:
                out_ += 1
                label_out['text'] = f'out: {out_}'
                tracking_paths = []
                new_tracking_paths = []
                frame_index_write = frame_index
                imsave(str(frame_index) + '.jpg', output_img)
        elif centropid(tracking_paths, default_centroid)[0] > xr + 65:
            if frame_index - frame_index_write > 40:
                in_ += 1
                label_in['text'] = f'in: {in_}'
                tracking_paths = []
                new_tracking_paths = []
                frame_index_write = frame_index
                imsave(str(frame_index) + '.jpg', output_img)
        cv2.circle(output_img, centropid(tracking_paths, default_centroid), 5, (0,0,255), -1)#drow centroid
        new_tracking_paths = new_tracking_paths[:1000]
    #Вход в блок 'if' после пропуска подходящего количества кадров
    if not frame_index % num_frames_jump:
        # Создание маски и вычерчивание окружностей
        mask = np.zeros_like(frame_gray[xl:xr, yb:yh])
        mask[:] = 255
        for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
            cv2.circle(mask, (x, y), 6, 0, -1)
        # Вычислим подходящие особые точки (признаки), подлежащие отслеживанию,
        # используя встроенную функцию с такими параметрами, как маска,
        # максимальное количество уrлов, уровень качества, минимальное расстояние
        # и размер блока.
        feature_points = cv2.goodFeaturesToTrack(frame_gray[xl:xr, yb:yh], #сложные настройки
                                                 mask = mask,
                                                 maxCorners = 500, 
                                                 qualityLevel = 0.1,
                                                 minDistance = 7, 
                                                 blockSize = 7) 
        # Проверка существования особых точек; если они
        # существуют, присоединить их к путям отслеживания
        if feature_points is not None:
            for x, y in np.float32(feature_points).reshape(-1, 2):
                tracking_paths.append([(x + xl, y + yb)])
    # обновляем переменные
    frame_index += 1
    frame_count['text'] = f'frame: {frame_index}'
    prev_gray = frame_gray
    cv2.rectangle(output_img, (xl, yh), (xr, yb), (255,0,0))# рисуем облость захвата
    tracking_paths = tracking_paths[:1000]
    # показать результат
    
    
    cv2image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream, cap, lmain, frame_index_write, num_frames_jump, tracking_paths, xl, xr, yb, yh,
                default_centroid, num_frames_to_track, frame_index, in_, out_, tracking_params, prev_gray) 


def open_run_movie():
    open_file = filedialog.askopenfilename()
    global lmain, flag, label_name_file
    label_name_file['text'] = f"file name: {open_file.split('/')[-1]}"
    if flag:
        lmain = Label(root)
        lmain.grid(row=1, column=0, columnspan=5)
        flag = False
    else:
        lmain.destroy()
        lmain = Label(root)
        lmain.grid(row=1, column=0, columnspan=5)
    
    cap = cv2.VideoCapture(open_file)
    args = get_param_for_start()
    video_stream(cap, lmain, *args)

    
button = Button(root, text='open file', command=open_run_movie)
button.grid(row=0, column=0)

root.mainloop()