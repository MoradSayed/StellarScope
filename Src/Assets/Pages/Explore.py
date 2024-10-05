import customtkinter as ctk
from CTK_Desert.Page_base_model import Page_BM
from CTK_Desert.Core import userChest as Chest
from CTK_Desert.Theme import theme
from CTK_Desert.utils import color_finder
import os, numpy as np, time
from PIL import Image
from OpenGL.GL import *
from .Engine_gl.Appgl import AppOgl
from collections import namedtuple
from tkinter import filedialog as fd

class Explore(Page_BM):
    def __init__(self):
        super().__init__(scrollable=False, 
                         start_func=self.on_start)
        self.parent = self.get_pf()

        fakeEventConst = namedtuple('event', ['x', 'y'])
        self.fake_event = fakeEventConst(x=0, y=0)

        self.f1 = ctk.CTkFrame(self.parent)
        self.f2 = ctk.CTkFrame(self.parent)
        self.f1.place(relx=0, rely=0, relwidth=0.7, relheight=1)
        self.f2.place(relx=0.7, rely=0, relwidth=0.3, relheight=1)
        
        width, height = self.f1.winfo_reqwidth(), self.f1.winfo_reqheight()
        self.app = AppOgl(self.f1, width, height)
        self.app.pack(fill="both", expand=True)
        self.key_binds()
        Chest.Window.bind("<FocusOut>", lambda e, state=4: self.app.handle_mouse(None, state, e.widget))

        # sec = ctk.CTkFrame(self.f2, fg_color=color_finder(self.f2))
        # ctk.CTkLabel(sec, text="Name", font=(theme.font, 18), text_color=theme.Ctxt, anchor="w").pack(fill="x", pady=(0,5))
        # ctk.CTkEntry(sec, font=(theme.font, 14), fg_color=theme.Csec, text_color=theme.Ctxt, textvariable=self.app.str_data).pack(fill="x")
        # sec.pack(fill="x", padx=10, pady=(10,0))
        # sec1 = ctk.CTkFrame(self.f2, fg_color=color_finder(self.f2))
        # ctk.CTkLabel(sec1, text="Distance", font=(theme.font, 18), text_color=theme.Ctxt, anchor="w").pack(fill="x", pady=(0,5))
        # ctk.CTkEntry(sec1, font=(theme.font, 14), fg_color=theme.Csec, text_color=theme.Ctxt, textvariable=ctk.StringVar(value="10 km")).pack(fill="x")
        # sec1.pack(fill="x", padx=10, pady=(10,0))

        self.general_frame = ctk.CTkTextbox(self.f2, font=(theme.font, 14))#, fg_color=theme.Csec, text_color=theme.Ctxt)
        self.general_frame.insert("1.0", text="General Info\n")
        self.general_frame.configure(state="disabled")  
        self.general_frame.pack(fill="both", expand=True, padx=0, pady=(0,0))

        name_placeHolder = "GL_Dome"    # self.widget_str.split(".")[-1][1:].capitalize()
        self.add_menu_button(os.path.join(os.path.join("Assets", "Pages", 'Engine_gl', "gfx", "icons8-camera-90.png")),
                              lambda: self.save_star_chart(f"{name_placeHolder}-{time.strftime('%H:%M:%S', time.gmtime()).replace(':', '_')}.png"), size=(35,35))
        # self.add_menu_button(os.path.join(r"C:\Users\Morad\Desktop\ES\Src\icons8-refresh-48.png"), lambda: self.change_side())
        # self.add_menu_button(os.path.join(r"C:\Users\Morad\Desktop\ES\Src\icons8-refresh-48.png"), lambda: self.raduis_factor_flip())
    
    def on_start(self):
        self.app.animate = 1
        # self.app.after(100, self.app.printContext)
        self.app.after(100, lambda: self.parent.bind("<Configure>", lambda e: self.app.on_resize()))

    def key_binds(self):
        Chest.Window.bind("w", self.app.handle_key)
        Chest.Window.bind("a", self.app.handle_key)
        Chest.Window.bind("s", self.app.handle_key)
        Chest.Window.bind("d", self.app.handle_key)
        
        Chest.Window.bind("<Control-c>", lambda e :self.change_pin())

        Chest.Window.bind("<Button-1>", lambda e, state=1: self.app.handle_mouse(None, state, e.widget))
        Chest.Window.bind("<Escape>",   lambda e, state=0: self.app.handle_mouse(None, state, e.widget))
        Chest.Window.bind("<Motion>",   lambda e : self.app.handle_mouse(e, widget=e.widget))
        Chest.Window.bind("<MouseWheel>", lambda e: self.app.handle_scroll_wheel(e.delta//120, e.widget))

    def save_star_chart(self, filename):
        self.app.render_text = 0
        self.update()
        width, height = self.f1.winfo_width(), self.f1.winfo_height()

        # Read pixels from the OpenGL buffer
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)    # Convert the pixel data to a numpy array
        pixels = np.flip(pixels, axis=0)    # Flip the image vertically

        image = Image.fromarray(pixels)
        """open a dialog to save the image"""
        fp = fd.askdirectory()
        image.save(rf"{fp}/{filename}")
        self.app.render_text = 1

    def change_pin(self):
        self.app.pin = not self.app.pin
        if self.app.pin:
            self.app.handle_mouse(self.fake_event, widget=self.app, override= 1)

    def change_side(self):
        self.app.in_out_state *= -1
        self.app.handle_mouse(self.fake_event, widget=self.app, override= 1)

    def raduis_factor_flip(self):
        self.app.raduis_factor *= -1