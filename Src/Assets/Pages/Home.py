import customtkinter as ctk
from CTK_Desert.Page_base_model import Page_BM
from CTK_Desert.Core import userChest as Chest
from CTK_Desert.Widgits import Banner, small_tabs, C_Widgits
import os, numpy as np

from .Engine_gl.CSVs_info import csv_reader

#* Don't pack Self.frame
class Home(Page_BM):
    def __init__(self):
        super().__init__(scrollable=True, )#start_func=self.on_start)
        self.parent = self.get_pf()

        self.setup_vars()

        Banner(self, self.parent, os.path.join("Assets", "Pages", 'Engine_gl', "gfx", "Mway.jpeg"), "The Milky Way!!", 
               "Explore the way we see our galaxy!", "Visit", lambda: None)     #! need to add a func
        
        custom_widgits = C_Widgits(self, self.parent) 
        planets_section = custom_widgits.section("Planets")
        self.sm_tabs = small_tabs(self, planets_section)
        self.after(200,self.construct)
        
    def construct(self):
        pt_names = self.get_planets(10)
        self.full_list.extend(pt_names)
        for name in pt_names:
            if name == "":
                continue
            self.sm_tabs.tab(f"{name}", image=os.path.join("Assets", "Pages", 'Engine_gl', "gfx", "planet.jpg"), 
                             button_icon=os.path.join("Assets","Pages","Engine_gl","gfx","icons8-right-arrow-64.png"), 
                             button_command= lambda ptn=name: self.tab_func(ptn), icon_size=(35,35))
        
    def setup_vars(self):
        self.csv_counter = 0
        self.full_list = []

    def get_planets(self, count):
        F_result = []
        result = list(Chest.MainPages["Explore"].app.ori_star_values[:, -1][self.csv_counter:count])
        for aList in result:
            if aList:
                F_result.append(aList[0])
            else:
                F_result.append("")
        self.csv_counter += count
        return F_result 
    
    def banner_func(self):
        Chest.MainPages["Explore"].app.pin_point = (0,0,0)
        Chest.MainPages["Explore"].app.scene.move_player_abs([-1630.0723/2, 531.7254/2, 245.21297/2])
        Chest.MainPages["Explore"].app.scene.spin_player_abs(theta=342, phi=-8)
        Chest.MainPages["Explore"].app.raduis_factor = -1
        Chest.MainPages["Explore"].change_side()

    def tab_func(self, pt_name):
        i = self.full_list.index(pt_name)
        stellar_data = Chest.MainPages["Explore"].app.ori_star_values[i]
        xyz_coords   = Chest.MainPages["Explore"].app.star_positions [i]
        print(xyz_coords)
        # print(pt_name, ">>", stellar_data)
        row = csv_reader.get_row(2, pt_name)

        Chest.MainPages["Explore"].app.pin_point = (xyz_coords[0], xyz_coords[1], xyz_coords[2])
        Chest.MainPages["Explore"].app.scene.move_player_abs([xyz_coords[0], xyz_coords[1], xyz_coords[2]+10])
        Chest.MainPages["Explore"].app.scene.spin_player_abs(theta=-stellar_data[0], phi=-stellar_data[1])
        Chest.MainPages["Explore"].app.scene.clamp_phi((10, 89))
        Chest.MainPages["Explore"].app.raduis_factor = 1
        Chest.MainPages["Explore"].change_side()
        Chest.MainPages["Explore"].general_frame.configure(state="normal")
        Chest.MainPages["Explore"].general_frame.insert("1.0", text=row)
        Chest.MainPages["Explore"].general_frame.configure(state="disabled")
        Chest.Switch_Page("Explore")

