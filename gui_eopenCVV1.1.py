import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, PhotoImage
from matplotlib.pyplot import imsave


class LKTrack:
    lk_params = dict(winSize=(11, 11),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, .003))
    
    feature_params = dict(maxCorners=500, qualityLevel=.01, minDistance=10)
    
    def __init__(self):
        self.root = Tk()
        self.flag_run = False
        self.frames = Label(self.root, text='frames: ')
        self.in_p = Label(self.root, text='in: 0')
        self.file_name = Label(self.root, text='file name: ')
        self.button_file = Button(self.root, text='open file', command=self.open_run_muvie)
        self.button_cum = Button(self.root, text='run on cum', command=self.open_run_cum)
        
        self.button_file.grid(row=0, column=0)
        self.button_cum.grid(row=0, column=1)
        self.file_name.grid(row=0, column=2)
        self.frames.grid(row=0, column=3)
        self.in_p.grid(row=0, column=4)
            
        self.track = []
        self.current_frame = 0
        self.people_in = 0
        self.build_online_rectangle = []
        self.cordinate_squade = [100, 200, 200, 300]
    
    def get_lmain(self):
        self.lmain = Label(self.root)
        self.lmain.bind('<Motion>', self.make_rectangle)
        self.lmain.grid(row=1, column=0, columnspan=5)
        
    def open_run_muvie(self):
        open_file = filedialog.askopenfilename()
        if open_file:
            if self.flag_run:
                self.lmain.destroy()
            self.flag_run = True
            self.get_lmain()
            self.cam = cv2.VideoCapture(open_file)
            self.file_name['text'] = 'file name: ' + open_file.split('/')[-1]
            self.current_frame = 0
            self.track = []
            self.detect_points()
    
    def open_run_cum(self):
        if self.flag_run:
            self.lmain.destroy()
        self.flag_run = True
        self.get_lmain()
        self.cam = cv2.VideoCapture(0)
        self.file_name['text'] = 'run on cam'
        self.current_frame = 0
        
    def detect_points(self):
        _, self.image =self.cam.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(
            self.gray[self.cordinate_squade[2]:self.cordinate_squade[3],
                      self.cordinate_squade[0]:self.cordinate_squade[1]], **self.feature_params)
        if features.__class__.__name__ != 'NoneType':
            features += np.array([[self.cordinate_squade[0], self.cordinate_squade[2]]], dtype=np.float32)  
            for x, y in features.reshape((-1, 2)):
                self.track.append([(x, y)])
        self.prev_gray = self.gray
        
    def track_points(self):
        _, self.image = self.cam.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        tmp = np.float32([tp[-1] for tp in self.track]).reshape(-1, 1, 2)
        features, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, tmp, None, **self.lk_params)
        features = [p for (st, p) in zip(status, features) if st]
        features = np.array(features).reshape((-1, 2))
        ndx = [i for (i, st) in enumerate(status) if not st]
        ndx.reverse()
        for i in ndx:
            self.track.pop(i)
        for i, f in enumerate(features):
            self.track[i].append(tuple(f))
        self.track = [i[-100:] for i in self.track]
        self.prev_gray = self.gray
        
    def point_is_move(self, point_track):#########################################??????????????????????
        if (abs(point_track[0][0] - point_track[-1][0]) > 30) and (point_track[-1][0] > self.cordinate_squade[1]):
            return True
        return False
    
    def centroid(self):##############################################################
        points = [p for p in self.track if self.point_is_move(p)]
        if len(points) > 10:
            mean_x = int(sum([p[-1][0] for p in points]) // len(points))
            mean_y = int(sum([p[-1][1] for p in points]) // len(points))
            cv2.circle(self.image, (mean_x, mean_y), 10, (200, 0, 0), -1)
            if mean_x > self.cordinate_squade[1] + 90:
                for i in points:
                    self.track.remove(i)
                self.people_in += 1
                self.in_p['text'] = f'in {self.people_in}'
                self.save_img()
        
    def draw_points(self):
        for index, (x, y) in enumerate((i[-1] for i in self.track)):
            color = (255, 0, 0) if self.point_is_move(self.track[index]) else (0, 255, 0)
            cv2.circle(self.image, (int(x), int(y)), 3, color, -1)
                
    def draw_rectangle(self):
        cv2.rectangle(self.image, 
                      (self.cordinate_squade[0], self.cordinate_squade[-1]), 
                      (self.cordinate_squade[1], self.cordinate_squade[2]), 
                      (0, 0, 255))
    
    def update_widget(self):
        self.current_frame += 1
        self.frames['text'] = f"frames: {self.current_frame}"      
    
    def update_window(self):
        img = Image.fromarray(self.image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
    
    def save_img(self):
        imsave(str(self.current_frame) + '.jpg', self.image)
        
    def make_rectangle(self, event):
        if event.state == 264:
            self.build_online_rectangle.append((event.x, event.y))
        elif self.build_online_rectangle:
            self.cordinate_squade.clear()
            self.cordinate_squade.append(min((self.build_online_rectangle[0][0], self.build_online_rectangle[-1][0])))
            self.cordinate_squade.append(max((self.build_online_rectangle[0][0], self.build_online_rectangle[-1][0])))
            self.cordinate_squade.append(min((self.build_online_rectangle[0][1], self.build_online_rectangle[-1][1])))
            self.cordinate_squade.append(max((self.build_online_rectangle[0][1], self.build_online_rectangle[-1][1])))
            self.build_online_rectangle.clear()
    
    def del_static_points(self):
        falg = True
        for point in self.track:
            if (point[-1][0] > self.cordinate_squade[1]) and not self.point_is_move(point):
                self.track.remove(point)
        
    def run(self):
        if self.flag_run:
            if (len(self.track) < 200):
                if not (self.current_frame % 200) and (not [i for i in self.track if self.point_is_move(i)]):
                    self.track.clear()
                self.detect_points()
            self.track_points()
            self.update_widget()
            self.draw_points()
            self.del_static_points()
            self.centroid()
            self.draw_rectangle()
            self.update_window()
        self.root.after(1, self.run) 
        return self.root

LKTrack().run().mainloop()