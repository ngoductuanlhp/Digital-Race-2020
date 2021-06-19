import numpy as np
import time

import modules.param as par
class SignTimer():
    def __init__(self):
        self.sign_array_0 = [[5],[0, 1],[-2],[5],[-2],[5],[-2],[5],[3]]
        self.sign_array_1 = [[5],[0],[4],[4],[4],[3]] 
        self.sign_array_2 = [[5],[1],[5],[-2],[5],[-2],[-2],[3]] 

        self.sign_array = self.sign_array_0
        self.timestamp = time.time()
        self.time_threshold = 0.8
        self.time_threshold_dup = 1.6
        self.timestamp_inter = time.time()

    def checkValidSign(self, blue_sign, red_sign):
        
        if par.timer_cursor == -1:
            if blue_sign in self.sign_array[par.timer_cursor + 1]:
                return True
            if red_sign in self.sign_array[par.timer_cursor + 1]:
                return True
            return False
        
        if par.timer_cursor == len(self.sign_array) - 1:
            if blue_sign in self.sign_array[par.timer_cursor]:
                return True
            if red_sign in self.sign_array[par.timer_cursor]:
                return True
            return False

        if blue_sign in self.sign_array[par.timer_cursor + 1] or blue_sign in self.sign_array[par.timer_cursor]:
            return True
        if red_sign in self.sign_array[par.timer_cursor + 1] or red_sign in self.sign_array[par.timer_cursor]:
            return True
        return False

    def update(self, blue_sign, red_sign, timestamp):
        if par.timer_cursor == len(self.sign_array) - 1:
            return False
            
        sign = -1
        if blue_sign >= 0 and blue_sign <3:
            sign = blue_sign
        elif red_sign >= 3 and red_sign <6:
            sign = red_sign
        else:
            return False

        if timestamp - self.timestamp < self.time_threshold:
            return False

        if par.timer_cursor == 0:
            if sign in self.sign_array[par.timer_cursor + 1]:
                if self.sign_array[par.timer_cursor + 1] == self.sign_array[par.timer_cursor]:
                    if timestamp - self.timestamp < self.time_threshold_dup:
                        return False
                par.timer_cursor += 1
                if sign == 0:
                    print("switch to map ngoai")
                    self.sign_array = self.sign_array_1
                    par.map_num = 1
                else:
                    print("switch to map trong")
                    self.sign_array = self.sign_array_2
                    par.map_num = 2

                self.timestamp = timestamp
            return True
        else:
            if sign in self.sign_array[par.timer_cursor + 1]:
                if self.sign_array[par.timer_cursor + 1] == self.sign_array[par.timer_cursor]:
                    if timestamp - self.timestamp < self.time_threshold_dup:
                        return False
                par.timer_cursor += 1
                if par.map_num == 1 and par.timer_cursor == 2:
                    par.dung_xe = True
                if par.map_num == 2 and par.timer_cursor == 5:
                    par.dung_xe = True
                self.timestamp = timestamp
            return True

    def update_interpolate(self, timestamp):
        if par.timer_cursor == len(self.sign_array) - 1:
            return False

        if -2 in self.sign_array[par.timer_cursor +1]:
            if timestamp -  self.timestamp_inter < 0.2:
                self.timestamp_inter = timestamp
                return False
            elif timestamp - self.timestamp_inter < 1.2:
                return False

            par.timer_cursor += 1
            
            if par.map_num == 1 and par.timer_cursor == 2:
                par.dung_xe = True
            if par.map_num == 2 and par.timer_cursor == 5:
                par.dung_xe = True

            self.timestamp_inter = timestamp
        return True

        