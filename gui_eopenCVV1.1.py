from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, VideoCapture, COLOR_BGR2GRAY, cvtColor, calcOpticalFlowPyrLK, circle, rectangle, goodFeaturesToTrack
import numpy as np
from tkinter import Tk, Label, Button, filedialog, PhotoImage
from PIL import ImageTk, Image
from matplotlib.pyplot import imsave


class LKTrack:
    _lk_params = dict(winSize=(11, 11),
                     maxLevel=2,
                     criteria=(TERM_CRITERIA_EPS|TERM_CRITERIA_COUNT, 10, .003))
    
    _feature_params = dict(maxCorners=500, qualityLevel=.01, minDistance=10)
    
    def __init__(self):
        self._root = Tk()
        self._flag_run = False
        self._frames = Label(self._root, text='frames: ')
        self._in_p = Label(self._root, text='in: 0')
        self._file_name = Label(self._root, text='file name: ')
        self._button_file = Button(self._root, text='open file', command=self._open_run_muvie)
        self._button_cum = Button(self._root, text='run on cum', command=self._open_run_cum)
        
        self._button_file.grid(row=0, column=0)
        self._button_cum.grid(row=0, column=1)
        self._file_name.grid(row=0, column=2)
        self._frames.grid(row=0, column=3)
        self._in_p.grid(row=0, column=4)
            
        self._build_online_rectangle = []
        self._cordinate_squade = self._load_squade()
    
    def _get_lmain(self):
        self._lmain = Label(self._root)
        self._lmain.bind('<Motion>', self._make_squade)
        self._lmain.grid(row=1, column=0, columnspan=5)
        
    def _open_run_muvie(self):
        open_file = filedialog.askopenfilename()
        if open_file:
            self._cam = VideoCapture(open_file)
            self._prepare_for_word()
            self._file_name['text'] = 'file name: ' + open_file.split('/')[-1]
    
    def _prepare_for_word(self):
    	if self._flag_run:
    		self._lmain.destroy()
    	self._flag_run = True
    	self._get_lmain()
    	self._current_frame = self._people_in = 0
    	self._track = []
    	self._detect_points()

    def _open_run_cum(self):
        self._cam = VideoCapture(0)
        self._prepare_for_word()
        self._file_name['text'] = 'run on cam'
     
    def _detect_points(self):
        _, self._image =self._cam.read()
        self._gray = cvtColor(self._image, COLOR_BGR2GRAY)
        features = goodFeaturesToTrack(
            self._gray[self._cordinate_squade[2]:self._cordinate_squade[3],
                      self._cordinate_squade[0]:self._cordinate_squade[1]], **self._feature_params)
        try:
            features += np.array([[self._cordinate_squade[0], self._cordinate_squade[2]]], dtype=np.float32)  
            for x, y in features.reshape((-1, 2)):
                self._track.append([(x, y)])
        except:
        	pass
        self._prev_gray = self._gray
        
    def _track_points(self):
        _, self._image = self._cam.read()
        self._gray = cvtColor(self._image, COLOR_BGR2GRAY)
        tmp = np.float32([tp[-1] for tp in self._track]).reshape(-1, 1, 2)
        features, status, _ = calcOpticalFlowPyrLK(self._prev_gray, self._gray, tmp, None, **self._lk_params)
        features = [p for (st, p) in zip(status, features) if st]
        features = np.array(features).reshape((-1, 2))
        ndx = [i for (i, st) in enumerate(status) if not st]
        ndx.reverse()
        for i in ndx:
            self._track.pop(i)
        for i, f in enumerate(features):
            self._track[i].append(tuple(f))
        self._track = [i[-100:] for i in self._track]
        self._prev_gray = self._gray
        
    def _point_is_move(self, point_track):
        if (abs(point_track[0][0] - point_track[-1][0]) > 30) and (point_track[-1][0] > self._cordinate_squade[1]):
            return True
        return False
    
    def _centroid(self):
        points = [p for p in self._track if self._point_is_move(p)]
        if len(points) > 10:
            mean_x = int(sum([p[-1][0] for p in points]) // len(points))
            mean_y = int(sum([p[-1][1] for p in points]) // len(points))
            circle(self._image, (mean_x, mean_y), 10, (200, 0, 0), -1)
            if mean_x > self._cordinate_squade[1] + 90:
                for i in points:
                    self._track.remove(i)
                self._people_in += 1
                self._in_p['text'] = f'in {self._people_in}'
                self._save_img()
        
    def _draw_points(self):
        for index, (x, y) in enumerate((i[-1] for i in self._track)):
            color = (255, 0, 0) if self._point_is_move(self._track[index]) else (0, 255, 0)
            circle(self._image, (int(x), int(y)), 3, color, -1)
                
    def _draw_rectangle(self):
        rectangle(self._image, 
                      (self._cordinate_squade[0], self._cordinate_squade[-1]), 
                      (self._cordinate_squade[1], self._cordinate_squade[2]), 
                      (0, 0, 255))
    
    def _update_widget(self):
        self._current_frame += 1
        self._frames['text'] = f"frames: {self._current_frame}"      
    
    def _update_window(self):
        img = Image.fromarray(self._image)
        imgtk = ImageTk.PhotoImage(image=img)
        self._lmain.imgtk = imgtk
        self._lmain.configure(image=imgtk)
    
    def _save_img(self):
        imsave(str(self._current_frame) + '.jpg', self._image)

    def _save_squade(self):
    	with open('cordinate_squade.txt', 'w') as file:
    		file.write(str(self._cordinate_squade))

    def _load_squade(self):
    	with open('cordinate_squade.txt') as file:
    		return eval(file.read())
        
    def _make_squade(self, event):
        if event.state == 264:
            self._build_online_rectangle.append((event.x, event.y))
        elif self._build_online_rectangle:
            self._cordinate_squade.clear()
            self._cordinate_squade.append(min((self._build_online_rectangle[0][0], self._build_online_rectangle[-1][0])))
            self._cordinate_squade.append(max((self._build_online_rectangle[0][0], self._build_online_rectangle[-1][0])))
            self._cordinate_squade.append(min((self._build_online_rectangle[0][1], self._build_online_rectangle[-1][1])))
            self._cordinate_squade.append(max((self._build_online_rectangle[0][1], self._build_online_rectangle[-1][1])))
            self._build_online_rectangle.clear()
            self._save_squade()
    
    def _del_static_points(self):
        falg = True
        for point in self._track:
            if (point[-1][0] > self._cordinate_squade[1]) and not self._point_is_move(point):
                self._track.remove(point)
        
    def run(self):
        if self._flag_run:
            if len(self._track) < 200:
                if not (self._current_frame % 200) and (not [i for i in self._track if self._point_is_move(i)]):
                    self._track.clear()
                self._detect_points()
            self._track_points()
            self._update_widget()
            self._draw_points()
            self._del_static_points()
            self._centroid()
            self._draw_rectangle()
            self._update_window()
        self._root.after(1, self.run) 
        return self._root


if __name__ == '__main__':
    LKTrack().run().mainloop()
