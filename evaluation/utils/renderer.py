import os
import numpy as np
import pyrender
import open3d as o3d

from typing import Tuple

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class Renderer():
    def __init__(self, height: int, width: int) -> None:
        """
        Initialize the renderer.
        :param height: height of the rendered image
        :param width: width of the rendered image
        """
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()

    def __call__(self, height: int, width: int, intrinsics: np.ndarray, 
                 pose: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render the mesh.
        :param height: height of the rendered image
        :param width: width of the rendered image
        :param intrinsics: camera intrinsics
        :param pose: camera pose
        :param mesh: mesh to render
        :return: rendered image, depth map
        """

        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)  # , self.render_flags)

    def fix_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Fix the pose of the camera.
        :param pose: camera pose
        :return: fixed camera pose
        """
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh: o3d.geometry.TriangleMesh) -> pyrender.Mesh:
        """
        Convert a mesh to a pyrender mesh.
        :param mesh: mesh to convert
        :return: pyrender mesh
        """
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self) -> None:
        """
        Delete the renderer.
        """
        self.renderer.delete()
        