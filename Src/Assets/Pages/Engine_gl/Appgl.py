from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLU import *
from pyopengltk import OpenGLFrame
import os, json, numpy as np, ctypes, pyrr, time, timeit
from math import sin, cos, sqrt, radians, pow, atan2, degrees
from PIL import Image
from typing import Union
from CTK_Desert.Core import userChest as Chest, ctk

from .Cam_controller import Scene
from .Fonting import Font, TextLine
from .Models.load import loadMesh
from .CSVs_info import csv_reader

class AppOgl(OpenGLFrame):
    def __init__(self, parent, width, height):
        super().__init__(master=parent)
        self.ffparent = parent
        self.width, self.height = width, height

        self.selected_object = None

        self.mouse_state = 0        # is the mouse active with the widget or not
        self.mouse_event_pass = 1   # blocks any calls if the cursor is going back to the middle of the screen
        self.str_data = ctk.StringVar(value="Earth")
        self.last_str = self.str_data.get()
        self.in_out_state : int[-1,1] = 1
        self.pin = 1                # Changed using ctrl key (pins and unpins the camera from a specific point)
        self.pin_point = (0,0,0)

        self.draw_lines = []
        self.waiting_list = []

        self.render_text = 1
        self.render_crosshair = 1

        self.raduis_factor = -1  #1 normal for normal browsing, -1 huge for global viewing 
        
        self.load_data("new_combined_file.json")
        get_row_func = lambda table, row_name: csv_reader.get_row(table, row_name)  #? request the data from the csv_reader

    def load_data(self, file_name: str):
        with open(os.path.join("Assets", "Cache", file_name), "r") as json_file:
            self.stars = json.load(json_file)
        self.star_keys = list(self.stars.keys())
        self.ori_star_values = np.array(list(self.stars.values()), dtype=object) # originally known as self.ori_coords
        self.star_coords = self.ori_star_values[:, :3].astype(dtype=np.float32)  # ra (deg), dec (deg), dist (parsecs)
        self.star_coords[:, :2] *= np.pi / 180                                   # ra (rad), dec (rad), dist (parsecs)
        # self.star_coords[:, 2] *= 5                                              # ra (rad), dec (rad), dist (km)
        self.temp_starsArray_len = len(self.star_coords)                         # Number of stars.     #!(Temp)

        """ Sun radius: 696,340 km
            parsec = 3.08567758*pow(10, 13) km
            Scale down everything so that a 1 sun radius star is 1 unit in the scene and 1 parsec is 44312801 units 
        """
        self.temp_radius_array = self.ori_star_values[:, 3:5]   # temp (K), rad (Rsun)
        self.temp_radius_array[:,0] = np.where(self.temp_radius_array[:,0] == None, 8989, self.temp_radius_array[:,0])
        self.temp_radius_array[:,1] = np.where(self.temp_radius_array[:,1] == None, 1   , self.temp_radius_array[:,1])
        colors = [4,3,2,1,0]
        conditions = [
            self.temp_radius_array[:, 0] >= 25000,                                              # Blue
            (self.temp_radius_array[:, 0] >= 10000) & (self.temp_radius_array[:, 0] < 25000),   # White
            (self.temp_radius_array[:, 0] >= 7500) & (self.temp_radius_array[:, 0] < 10000),    # Yellow
            (self.temp_radius_array[:, 0] >= 5000) & (self.temp_radius_array[:, 0] < 7500),     # Orange
            self.temp_radius_array[:, 0] < 5000,                                                # Red
            ]
        self.temp_radius_array[:, 0] = np.select(conditions, colors, default='Unknown')
        self.temp_radius_array= self.temp_radius_array.astype(dtype=np.float32)
        
        """used to calculate the cartesian coordinates of the stars"""
        r = self.star_coords[:, 2]
        theta = self.star_coords[:, 0]
        phi = self.star_coords[:, 1]
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(phi)
        self.star_positions = np.stack([x, y, z], axis=-1)                       # x, y, z positions of the stars
      
    def initgl(self):
        """Initialize OpenGL states when the frame is created"""        
        glViewport(0, 0, self.width, self.height)
        self.scene = Scene()
        self.rederer = GraphicsEngine(self)

        """vars used in calculating the frame rate"""
        self.lastTime = time.time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0  # time.time() - self.lastTime
        self.framerate = 0

    def on_resize(self):
        """Handle window resizing: Update the OpenGL viewport and perspective with the new dimensions"""
        self.width, self.height = self.ffparent.winfo_width(), self.ffparent.winfo_height()
        
        glViewport(0, 0, self.width, self.height)
        
        glUseProgram(self.rederer.shader)
        projection_transform = pyrr.matrix44.create_perspective_projection(
            45, self.width/self.height, 0.1, 100000, dtype=np.float32)
        glUniformMatrix4fv(
            glGetUniformLocation(self.rederer.shader, "projection"), 
            1, GL_FALSE, projection_transform)        

    def redraw(self):
        """Render a single frame"""
        self.scene.update(self.frameTime/16.7)
        self.rederer.render(self.scene)
        self.calculateFramerate()        

    def handle_key(self, event):
        # match event.keysym:
        #     case 'w':
        #         self.handle_scroll_wheel(1)
        #     case 's':
        #         self.handle_scroll_wheel(-1)
        pass

    def handle_mouse(self, event, state=None, widget=None, override=0):
        if widget==self:
            if state == 1:
                if self.mouse_state == 0:
                    """Get the Mouse into the widget"""
                    self.ffparent.configure(cursor='none')
                    self.ffparent.event_generate('<Motion>', warp=True, x=self.ffparent.winfo_width()/2, y=self.ffparent.winfo_height()/2)
                else:
                    """Select a Star"""
                    self.selected_object = self.check_intersect()
                    if self.selected_object:
                        self.waiting_list.append(self.star_positions[self.star_keys.index(self.selected_object)].tolist())
                        if len(self.waiting_list) == 2:
                            self.draw_lines.extend([self.waiting_list])
                            self.waiting_list = []
                        # print("main", self.draw_lines, ",", "waiting", self.waiting_list, ">>", self.selected_object)
                self.mouse_state = state

            elif state == 0 and self.mouse_state:
                """Get the Mouse out of the widget"""
                self.ffparent.configure(cursor='arrow')
                self.mouse_state = state

            elif state == None:
                """Mouse motion"""
                if (self.mouse_state and self.mouse_event_pass) or override:
                    self.mouse_event_pass = 0
                    midX, midY = self.ffparent.winfo_width()/2, self.ffparent.winfo_height()/2
                    x, y = event.x, event.y
                    if not override:
                        self.pin_to_point(midX-x,midY-y, self.pin_point)
                        self.ffparent.event_generate('<Motion>', warp=True, x=midX, y=midY)
                        self.update()
                    else:
                        self.pin_to_point(0, 0, self.pin_point)
                    self.mouse_event_pass = 1

        elif state == 4 and self.mouse_state and widget == Chest.Window:
            """Window out of focus"""
            self.ffparent.configure(cursor='arrow')
            self.mouse_state = 0
    
    def check_intersect(self):
        """Check if the spheres are in the camera's field of view"""
        camera_fov = np.radians(45)
        tolerance = np.radians(5)
        vectors_to_spheres = self.star_positions - self.scene.player.position                           # Calculate vectors from camera to spheres
        camera_forward_norm = self.scene.player.forwards / np.linalg.norm(self.scene.player.forwards)   # Normalize the forward vector and vectors to spheres
        vectors_to_spheres_norm = vectors_to_spheres / np.linalg.norm(vectors_to_spheres, axis=1)[:, np.newaxis]        
        cos_angles = np.einsum('ij,j->i', vectors_to_spheres_norm, camera_forward_norm)                 # dot product of the normalized vectors (2d array * 1d array) -> 1d array
        angles = np.arccos(cos_angles)                                                                  # Calculate the cosine of the angles
        filtered_indices = np.where(angles <= (camera_fov / 2 + tolerance))
        filtered_spheres = self.star_positions[filtered_indices]                                        # Get angles and compare to FOV + tolerance
        
        """Cast the ray and check for intersections"""
        intersections = []
        for i, sph_pos in enumerate(filtered_spheres):    #? enumrate to keep the index
            intersect_result = ray_sphere_intersection(self.scene.player.position, self.scene.player.forwards, sph_pos, 0.5)#*self.temp_radius_array[filtered_indices[0][i]][1])
            if intersect_result:
                intersections.append(intersect_result)
        result = min(intersections, default=None)
        if result:
            sphere_name = self.star_keys[filtered_indices[0][np.where(filtered_spheres==result[1])[0][0]]]
            return sphere_name
        return None

    def pin_to_point(self, dx, dy, point: tuple[int, int]): 
        """Responsable for the camera movement using the mouse (either pinned or not)"""
        if self.pin:
            rate = self.frameTime / 16.7
            theta_increment = rate * dx *0.5
            phi_increment = rate * dy   *0.5
            self.scene.spin_player(theta_increment, phi_increment)

            distance = sqrt(pow(point[0]-self.scene.player.position[0], 2) + 
                            pow(point[1]-self.scene.player.position[1], 2) + 
                            pow(point[2]-self.scene.player.position[2], 2)) * (self.in_out_state*1)
            theta = radians((self.scene.player.theta+180)%360)
            phi =   -radians(self.scene.player.phi)
            x = point[0] + distance * cos(theta) * cos(phi)
            y = point[1] + distance * sin(theta) * cos(phi)
            z = point[2] + distance * sin(phi)
            self.scene.player.position[:] = np.array([x, y, z], dtype=np.float32)
        else:
            rate = self.frameTime / 16.7
            theta_increment = rate * dx
            phi_increment = rate * dy
            self.scene.spin_player(theta_increment, phi_increment)

    def is_in_out(self, point):
        """
        dx = point[0] - self.scene.player.position[0]  
        dy = point[1] - self.scene.player.position[1]  
        theta = (degrees(atan2(dy, dx))   + 360) % 360  # angle in degrees between the two points  
        if abs(theta - self.scene.player.theta) < 170:  
            return 1  
        return -1  
        """
    
    def handle_scroll_wheel(self, amount, widget):
        if widget == self:
            """Handle the mouse wheel event"""
            dpos = [
                self.frameTime/16.7 * np.cos(np.deg2rad(self.scene.player.theta)) * np.cos(np.deg2rad(self.scene.player.phi)) * amount*3,
                self.frameTime/16.7 * np.sin(np.deg2rad(self.scene.player.theta)) * np.cos(np.deg2rad(self.scene.player.phi)) * amount*3,
                self.frameTime/16.7 * np.sin(np.deg2rad(self.scene.player.phi  )) * amount*3
            ]
            self.scene.move_player(dpos)

    def calculateFramerate(self):
        """Calculate the framerate"""
        self.currentTime = time.time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            self.framerate = max(1, int(self.numFrames/delta))
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0/max(1,self.framerate))
        self.numFrames += 1

class ShpereMesh:
    def __init__(self):
        # x, y, z, s, t, nx, ny, nz
        sphere_vert = loadMesh(os.path.join("Assets", "Pages", "Engine_gl", "Models", "Sphere.obj"))
        self.sphere_vert_count = len(sphere_vert) // 8
        sphere_vert = np.array(sphere_vert, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_vert.nbytes, sphere_vert, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

class CubeMesh:
    def __init__(self):
        # x, y, z, s, t, nx, ny, nz
        cube_vert = loadMesh(os.path.join("Assets", "Pages", "Engine_gl", "Models", "Cube.obj"))
        self.cube_vert_count = len(cube_vert) // 8
        cube_vert = np.array(cube_vert, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, cube_vert.nbytes, cube_vert, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

class Sphere:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class Material:

    def __init__(self, filepaths: list):
        self.textures = []
        for fp in filepaths:
            self.generate(fp)

    def generate(self, filepath):
        texture = glGenTextures(1)
        self.textures.append(texture)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        with Image.open(filepath) as image: 
            image_width, image_height = image.size
            image = image.convert("RGBA")
            img_data = image.tobytes("raw", "RGBA")
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self, indexes=[0,]):
        for i in indexes:
            glActiveTexture(eval(f"GL_TEXTURE{i}"))
            glBindTexture(GL_TEXTURE_2D,self.textures[i])

    def destroy(self):
        for texture in self.textures:
            glDeleteTextures(1, (texture,))

class GraphicsEngine:
    def __init__(self, app:AppOgl):
        self.app = app
        self.shpere_mesh = ShpereMesh()
        thepath= os.path.join("Assets", "Pages", "Engine_gl", "gfx", "Stars")
        self.texture = Material([os.path.join(thepath, "Red.png"   ), 
                                 os.path.join(thepath, "Orange.png"),
                                 os.path.join(thepath, "Yellow.png"),
                                 os.path.join(thepath, "White.png" ),
                                 os.path.join(thepath, "Blue.png"  )])

        thepath= os.path.join("Assets", "Pages", "Engine_gl", "Shaders")
        self.shader = self.prep_shaders(os.path.join(thepath, "vert.glsl"), 
                                        os.path.join(thepath, "frag.glsl"))
        self.render_sphere_setup()
        self.line_shader = self.prep_shaders(os.path.join(thepath, "line_vert.glsl"), 
                                             os.path.join(thepath, "line_frag.glsl"))
        self.setup_line_rendering()
        
        self.textShader = self.prep_shaders(os.path.join(thepath, "text_vert.glsl"),
                                            os.path.join(thepath, "text_frag.glsl"))
        glUseProgram(self.textShader)
        glUniform1i(glGetUniformLocation(self.textShader, "material"), 0)
        self.font = Font()
        self.fps_label = TextLine(app.str_data.get(), self.font, (-0.9, 0.9), (0.05, 0.05))

    def prep_shaders(self, vert_path, frag_path):
        """Prepare the shaders"""
        with open(vert_path,'r') as f:
            vertex_src = f.readlines()

        with open(frag_path,'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader
    
    def render_sphere_setup(self):
        glUseProgram(self.shader)
        for i in range(len(self.texture.textures)):
            glUniform1i(glGetUniformLocation(self.shader, f"imageTextures[{i}]"), i)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            45, self.app.width/self.app.height, 0.1, 100000, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"), 
            1, GL_FALSE, projection_transform
        )        

        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.radius_factor_loc = glGetUniformLocation(self.shader, "size_op")

        #? this data isn't actually used, it's just created to allocate that space
        self.sphereTransforms = np.array([      
            pyrr.matrix44.create_identity(dtype=np.float32)
            for i in range(self.app.temp_starsArray_len)
        ], dtype=np.float32)
        self.sphereTransformVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphereTransformVBO)
        glBufferData(GL_ARRAY_BUFFER, self.sphereTransforms.nbytes, self.sphereTransforms, GL_STATIC_DRAW)
        #? #######################################################
        glBindVertexArray(self.shpere_mesh.vao)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(16))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(32))
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(48))
        glVertexAttribDivisor(2, 1)
        glVertexAttribDivisor(3, 1)
        glVertexAttribDivisor(4, 1)
        glVertexAttribDivisor(5, 1)

        self.textureindex_radius_VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.textureindex_radius_VBO)
        # print(self.app.temp_radius_array)
        glBufferData(GL_ARRAY_BUFFER, self.app.temp_radius_array.nbytes, self.app.temp_radius_array, GL_STATIC_DRAW)
        glEnableVertexAttribArray(6)
        glVertexAttribPointer(6, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glVertexAttribDivisor(6, 1)

        self.texture.use([0,1,2,3,4])    # Texture: needs some setup in the 
        glBindVertexArray(self.shpere_mesh.vao)
        self.model_matrix_counter = 0
        for sph_loc in self.app.star_coords:
            self.modelMatrix(sph_loc)
        self.model_matrix_counter = 0

    def setup_line_rendering(self):
        glUseProgram(self.line_shader)
        self.line_vao = glGenVertexArrays(1)
        self.line_vbo = glGenBuffers(1)

        glBindVertexArray(self.line_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            45, self.app.width/self.app.height, 0.1, 100000, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.line_shader, "projection"), 
            1, GL_FALSE, projection_transform
        ) 
        self.Line_viewMatrixLoc = glGetUniformLocation(self.line_shader, "view")
        
        self.collect_new_lines()

        glBufferData(GL_ARRAY_BUFFER, self.line_points.nbytes, self.line_points, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def collect_new_lines(self):
        line_points = []
        for line in self.app.draw_lines:
            line_points.extend(line[0])  # Start point
            line_points.extend(line[1])  # End point

        self.line_points = np.array(line_points, dtype=np.float32)

    def render(self, scene:Scene):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        """Scene rendering"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glUseProgram(self.shader)
        glUniform1f(self.radius_factor_loc, self.app.raduis_factor)
        view_transform = pyrr.matrix44.create_look_at(
            eye = scene.player.position,
            target = scene.player.position + scene.player.forwards,
            up = scene.player.up, dtype=np.float32)
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)
        
        self.texture.use([0,1,2,3,4])    # Texture: needs some setup in the 
        glBindVertexArray(self.shpere_mesh.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.sphereTransformVBO)
        glBufferData(GL_ARRAY_BUFFER, self.sphereTransforms.nbytes, self.sphereTransforms, GL_STATIC_DRAW)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.shpere_mesh.sphere_vert_count, self.app.temp_starsArray_len)       

        """Line rendering"""
        if self.app.draw_lines:
            glUseProgram(self.line_shader)
            glUniformMatrix4fv(self.Line_viewMatrixLoc, 1, GL_FALSE, view_transform)
            glBindVertexArray(self.line_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
            self.collect_new_lines()
            glBufferData(GL_ARRAY_BUFFER, self.line_points.nbytes, self.line_points, GL_STATIC_DRAW)
            glLineWidth(3.0)
            glDrawArrays(GL_LINES, 0, len(self.line_points) // 3)

        """Text rendering"""
        if self.app.render_text:
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            glUseProgram(self.textShader)
            disp_string = ""
            self.font.use()
            disp_string += f"theta: {self.app.scene.player.theta:.2f}, phi: {self.app.scene.player.phi:.2f}"   #* No.1: Theta and Phi
            # disp_string += f"FPS: {self.app.framerate}"                                                        #* No.2: FPS
            self.fps_label.build_text(disp_string, self.font, self.app.render_crosshair)
            # if self.app.str_data.get() != self.app.last_str:                                                                        #* No.3: Custom String
                # disp_string += self.app.str_data.get()
                # self.app.last_str = self.app.str_data.get()
                # self.fps_label.build_text(disp_string, self.font)
            self.fps_label.draw()

    def modelMatrix(self, sph, index=None):
        if index:
            position = np.array(
                [sph[2] * cos(sph[0]) * cos(sph[1]),
                 sph[2] * sin(sph[0]) * cos(sph[1]),
                 sph[2] * sin(sph[1])            ] , dtype=np.float32)
            self.app.star_positions[index] = position
        else:
            position = self.app.star_positions[self.model_matrix_counter]
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
        model_transform = pyrr.matrix44.multiply(m1=model_transform,
            m2=pyrr.matrix44.create_from_eulers(np.radians([0,0,0]), dtype=np.float32))
        model_transform = pyrr.matrix44.multiply(m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(position, dtype=np.float32))
        if index:
            self.sphereTransforms[index] = model_transform
            return
        self.sphereTransforms[self.model_matrix_counter] = model_transform
        self.model_matrix_counter+=1

def ray_sphere_intersection(ray_origin, ray_dir, sphere_center, sphere_radius): # Ray-Sphere Intersection Function
    # Calculate quadratic equation coefficients
    oc = ray_origin - sphere_center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius ** 2
    
    # Solve the discriminant
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None  # No intersection
    else:
        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        
        # Return the closest positive intersection
        if t1 > 0:
            return t1, sphere_center
        elif t2 > 0:
            return t2, sphere_center
        else:
            return None
        
