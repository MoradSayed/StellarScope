from OpenGL.GL import *
import numpy as np, random
from typing import Optional

class Player:  # holds the camera position and orientation
    def __init__(self, position, theta=0, phi=0):
        self.position = np.array(position, dtype=np.float32)
        self.theta = theta  # horizontal angle (declination)
        self.phi = phi    # vertical angle   (right ascension)
        self.phi_range = (-89, 89)
        self.update_vectors()

    def update_vectors(self):
        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi)),
            ]
        )

        globalUp = np.array([0, 0, 1], dtype=np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)

class Scene:    # updates the player values and the scenes
    def __init__(self):
        from .Appgl import Sphere
        # self.spheres = [
        #     # Sphere(
        #     #     position = [6,0,0], # [random.uniform(-10, 10) for x in range(3)]
        #     #     eulers   = [0,0,0]  # [random.uniform(0, 360) for x in range(3)]
        #     # )
        # ]
        self.player = Player(position=[-1630.0723/2, 531.7254/2, 245.21297/2], theta=342, phi=-8)
        # self.player = Player(position=[-1,0,0])
        
    def update(self, rate):     #! needs a look
        """Update objects in the scene"""
        # for sphere in self.spheres:
        #     sphere.eulers[1] = ((0.25 * rate)+(sphere.eulers[1]))%360
        pass

    def move_player(self, dPos):
        """Relative Movement"""
        dPos = np.array(dPos, dtype=np.float32)
        self.player.position += dPos

    def move_player_abs(self, Pos):
        """Absolute Movement"""
        Pos = np.array(Pos, dtype=np.float32)
        self.player.position = Pos

    def spin_player(self, dTheta, dPhi):
        """Relative Rotation"""
        self.player.theta = (self.player.theta+dTheta)%360

        self.player.phi = min(
            self.player.phi_range[1], max(self.player.phi_range[0], self.player.phi + dPhi)
        )

        self.player.update_vectors()

    def spin_player_abs(self, theta, phi):
        """Absolute Rotation"""
        self.player.theta = theta%360
        self.player.phi = min(
            self.player.phi_range[1], max(self.player.phi_range[0], phi)
        )
        self.player.update_vectors()

    def clamp_phi(self, range: Optional[tuple[int, int]]):
        """Clamp the vertical angle"""
        self.player.phi_range = range

