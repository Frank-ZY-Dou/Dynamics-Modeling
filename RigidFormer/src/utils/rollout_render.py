"""Fast physically based rendering of rigid-body rollouts, offscreen on the GPU (Filament, through Open3D).

Dynamics use the objects' collision meshes; pictures use their visual meshes. The rigid pose of every object in every
frame is fitted (Kabsch) to its collision-mesh vertices, the model's rigid prediction, and applied to the object's visual
mesh (KuBasic visual_geometry.obj, the meshes the MOVi videos were rendered with; same frame as the collision mesh).

The scene: a checker floor (1 m tiles), light (default) or dark, image-based lighting, a key light with soft shadows,
ambient occlusion and filmic tone mapping; physically based materials (a fixed palette; metal objects metallic). A fixed
camera frames where the objects move. Given the times of the key poses, a video samples the motion in real time at
its frame rate, interpolating between key poses. Frames are rendered at `supersample` times the output size, averaged
down and piped to ffmpeg (H.264). Nothing but the scene is drawn: no text. Without a GPU, EGL_PLATFORM=surfaceless
renders the same scene with Mesa's software rasterizer (llvmpipe), more slowly.

    meshes = load_render_meshes('data_generation/objects', shapes)
    poses = [fit_poses(size * collision_vertices(mesh_dir, shape), frames)[0] ...]   # per object: (T, 4, 4)
    get_renderer().render(out_path, [meshes[s] for s in shapes], sizes, metal, poses, fps, ghost=None)
"""
import functools
import os
import subprocess

import numpy as np

# object colours (sRGB): blue, orange, red, teal, green, yellow, purple, pink, brown, grey
PALETTE = ('#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac')


def linear_rgb(colour):
    """Linear RGB of an sRGB hex colour (materials take linear colours)."""
    c = np.array([int(colour[k:k + 2], 16) for k in (1, 3, 5)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# Studio looks. A checker floor in light or dark tones, image-based light for soft fill and reflections, one warm key
# light from above, left of the camera, for form and soft contact shadows; filmic tone mapping. Plastic objects with a
# clear coat, metal ones metallic. Floor albedos give rendered tones of about 221 / 209 (light) and 35 / 24 (dark).
LOOKS = {
    'light': dict(background=(0.86, 0.867, 0.87), checker=((0.97, 0.985, 1.0), (0.74, 0.76, 0.79)), floor_roughness=0.95,
                  floor_reflectance=0.2, ibl_intensity=33000, key_intensity=130000),
    'dark': dict(background=(0.1, 0.104, 0.112), checker=((0.04, 0.042, 0.046), (0.026, 0.027, 0.03)),
                 floor_roughness=0.85, floor_reflectance=0.12, ibl_intensity=30000, key_intensity=120000),
}
for _look in LOOKS.values():
    _look.update(ibl='default', key=(1.0, 0.957, 0.902), roughness=0.45, reflectance=0.5, clearcoat=0.5,
                 clearcoat_roughness=0.12, metal_roughness=0.3)
KEY_DIRECTION = (-0.02, 0.42, -0.91)  # key light travel direction (world, z up): from above, left of the camera
FLOOR_SIZE = 2000.0                # m
TILE = 1.0                         # floor tile (m)
GHOST = (0.55, 0.57, 0.62, 0.3)    # translucent ground truth
POINT = (0.93, 0.94, 0.95)         # style 'solid': colour of the drawn surface points, linear RGB
SEE_THROUGH = 0.55                 # style 'points': opacity of the object meshes around their points
GLASS_TINT = 0.1                   # style 'points': mesh colour, the object's blended this far towards white
POINT_SHADE = 0.75                 # style 'points': point colour, the object's darkened by this factor
ADDED_TINT = 0.6                   # style 'points': added-point colour, the object's blended this far towards white
ADDED_RADIUS = 0.6                 # radius of the added (completion) points, relative to the observed points'
VIEW = (1.0, -1.25, 1.35)          # camera direction (from the target)
FOV = 30.0                         # vertical field of view (degrees)


def kabsch(X0, X):
    """Rigid motion (R, t) with X ~ R X0 + t."""
    c0, c = X0.mean(0), X.mean(0)
    U, _, Vt = np.linalg.svd((X0 - c0).T @ (X - c))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, c - R @ c0


def fit_poses(reference, frames):
    """Rigid poses taking `reference` (V, 3), an object's vertices in its own frame, to frames[t] (V, 3): (T, 4, 4)
    and the largest distance of a fitted vertex from its frame's vertex."""
    X0, X = np.asarray(reference, np.float64), np.asarray(frames, np.float64)
    c0, c = X0.mean(0), X.mean(1)
    U, _, Vt = np.linalg.svd(np.einsum('vi,tvj->tij', X0 - c0, X - c[:, None]))
    V, Ut = np.transpose(Vt, (0, 2, 1)), np.transpose(U, (0, 2, 1))
    D = np.tile(np.eye(3), (len(X), 1, 1))
    D[:, 2, 2] = np.sign(np.linalg.det(V @ Ut))
    R = V @ D @ Ut
    out = np.tile(np.eye(4), (len(X), 1, 1))
    out[:, :3, :3], out[:, :3, 3] = R, c - R @ c0
    resid = float(np.linalg.norm(np.einsum('tij,vj->tvi', R, X0) + out[:, None, :3, 3] - X, axis=-1).max())
    return out, resid


def poses_from_vertices(reference, frames):
    return fit_poses(reference, frames)[0]


def interpolate_poses(poses, times, t):
    """Poses (len(t), 4, 4) at times t from key poses (K, 4, 4) at increasing `times`: positions interpolated
    linearly, rotations along the shortest arc (spherical linear interpolation); held outside the key times."""
    from scipy.spatial.transform import Rotation
    poses, times, t = np.asarray(poses, np.float64), np.asarray(times, np.float64), np.asarray(t, np.float64)
    if len(times) == 1:
        return np.repeat(poses, len(t), axis=0)
    k = np.clip(np.searchsorted(times, t, side='right') - 1, 0, len(times) - 2)
    a = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)
    R0, R1 = poses[k, :3, :3], poses[k + 1, :3, :3]
    step = Rotation.from_matrix(np.transpose(R0, (0, 2, 1)) @ R1).as_rotvec()
    out = np.tile(np.eye(4), (len(t), 1, 1))
    out[:, :3, :3] = R0 @ Rotation.from_rotvec(a[:, None] * step).as_matrix()
    out[:, :3, 3] = (1 - a)[:, None] * poses[k, :3, 3] + a[:, None] * poses[k + 1, :3, 3]
    return out


@functools.lru_cache(maxsize=None)
def collision_vertices(mesh_dir, shape):
    """Vertices of <mesh_dir>/<shape>/collision_geometry.obj in file order: the order of an object's vertices in the
    data (unit size, the object's frame)."""
    import trimesh
    return np.asarray(trimesh.load(os.path.join(mesh_dir, shape, 'collision_geometry.obj'), process=False).vertices,
                      np.float64)


@functools.lru_cache(maxsize=None)
def load_obj(path):
    """(vertices, normals, faces) of a Wavefront OBJ file; polygons are triangulated as fans. One vertex per distinct
    (position, normal) pair of the face corners, so a hard edge in the file stays hard. The normals are the file's
    directions recomputed as face-area-weighted means over the faces that share each vertex: the large flat faces
    of the bevelled shapes (two triangles per cube face) then shade flat, where the file's normals, tilted toward
    the bevels, draw a highlight line across them; curved surfaces stay smooth."""
    pos, nrm, corners, faces = [], [], {}, []
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                pos.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('vn '):
                nrm.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                ids = []
                for c in line.split()[1:]:
                    parts = c.split('/')
                    if len(parts) < 3 or not parts[2]:
                        raise ValueError(f'{path}: a face corner without a normal ({c!r})')
                    key = (int(parts[0]) - 1, int(parts[2]) - 1)
                    ids.append(corners.setdefault(key, len(corners)))
                faces += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
    keys = np.array(list(corners), np.int64)
    V = np.asarray(pos, np.float64)[keys[:, 0]]
    file_normals = np.asarray(nrm, np.float64)[keys[:, 1]]
    F = np.asarray(faces, np.int64)
    cross = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])          # length: twice the face area
    N = np.zeros_like(V)
    for j in range(3):
        np.add.at(N, F[:, j], cross)
    length = np.linalg.norm(N, axis=1, keepdims=True)
    if (length[:, 0] == 0).any() or ((N * file_normals).sum(1) <= 0).any():
        raise ValueError(f'{path}: vertex normals undefined or against the file normals (face winding)')
    return V, N / length, F


def point_spheres(points, radius, resolution=4):
    """One mesh of small spheres of `radius` centred at `points` (N, 3): (vertices, normals, faces)."""
    import open3d as o3d
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    sphere.compute_vertex_normals()
    V0, N0, F0 = np.asarray(sphere.vertices), np.asarray(sphere.vertex_normals), np.asarray(sphere.triangles)
    P = np.asarray(points, np.float64)
    V = (P[:, None, :] + radius * V0[None]).reshape(-1, 3)
    N = np.broadcast_to(N0, (len(P),) + N0.shape).reshape(-1, 3)
    F = (F0[None] + (np.arange(len(P)) * len(V0))[:, None, None]).reshape(-1, 3)
    return V, N, F


def load_render_meshes(mesh_dir, shapes):
    """{shape: (vertices, normals, faces)} of <mesh_dir>/<shape>/visual_geometry.obj."""
    out = {}
    for s in set(shapes):
        path = os.path.join(mesh_dir, s, 'visual_geometry.obj')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path}: no visual mesh for shape {s!r}')
        out[s] = load_obj(path)
    return out


def frame_camera(poses, radii, aspect, fov=FOV, view=VIEW, margin=1.04):
    """(target, eye) of a fixed camera that keeps every object in view over the whole clip: each object, a sphere of
    its radius around its centre in every frame, must fit in the field of view seen from `view`; the distance is the
    smallest that fits all of them, times `margin`."""
    centres = np.stack([p[:, :3, 3] for p in poses], axis=1).reshape(-1, 3)        # (T * n, 3)
    r = np.tile(np.asarray(radii, np.float64), len(poses[0]))
    lo, hi = (centres - r[:, None]).min(0), (centres + r[:, None]).max(0)
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.5 * max(hi[2], 0.0) * 0.6])
    f = -np.asarray(view, np.float64) / np.linalg.norm(view)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    rel = centres - target
    x, y, z = rel @ right, rel @ up, rel @ f
    tv = np.tan(np.radians(fov / 2))
    th = tv * aspect
    # a sphere fits when its centre, widened by the radius across and brought nearer by it, does
    dist = max(float(((np.abs(x) + r) / th - z + r).max()), float(((np.abs(y) + r) / tv - z + r).max()), 1.0) * margin
    return target, target - f * dist


class RolloutRenderer:
    def __init__(self, width=1280, height=720, supersample=2, floor='light', look=None):
        import open3d as o3d
        from open3d.visualization import rendering
        if floor not in LOOKS:
            raise ValueError(f'floor must be one of {sorted(LOOKS)}, got {floor!r}')
        self.o3d, self.rendering = o3d, rendering
        self.width, self.height, self.ss, self.floor_tone = width, height, supersample, floor
        self.look = dict(LOOKS[floor], **(look or {}))
        L = self.look
        self.renderer = rendering.OffscreenRenderer(width * supersample, height * supersample)
        scene = self.renderer.scene
        scene.set_background([*L['background'], 1.0])
        if not scene.scene.set_indirect_light(os.path.join(os.path.dirname(o3d.__file__), 'resources', L['ibl'])):
            raise RuntimeError(f"could not load the image-based light {L['ibl']!r} of Open3D's resources")
        scene.scene.enable_indirect_light(True)
        scene.scene.set_indirect_light_intensity(L['ibl_intensity'])
        scene.scene.set_sun_light(list(KEY_DIRECTION), list(L['key']), L['key_intensity'])
        scene.scene.enable_sun_light(True)
        view = scene.view
        view.set_shadowing(True, rendering.View.ShadowType.PCF)
        view.set_ambient_occlusion(True)
        view.set_antialiasing(True)
        view.set_post_processing(True)
        view.set_color_grading(rendering.ColorGrading(rendering.ColorGrading.Quality.ULTRA,
                                                      rendering.ColorGrading.ToneMapping.ACES_LEGACY))
        self.floor = self._floor()
        self._prewarm()

    def _prewarm(self, max_frames=300):
        """The engine compiles shader variants on first use and leaves objects out of the frames until they are
        ready. Draw the floor with one object of each material (plastic, metal, translucent) on it until all three
        show against the empty floor."""
        o3d, scene = self.o3d, self.renderer.scene
        box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        box.compute_vertex_normals()
        V, N, F = np.asarray(box.vertices) + [-0.5, -0.5, 0.0], np.asarray(box.vertex_normals), np.asarray(box.triangles)
        scene.clear_geometry()
        scene.add_geometry('floor', *self.floor)
        self.renderer.setup_camera(FOV, [0, 0, 0.5], [0, -12, 4.5], [0, 0, 1])
        for _ in range(3):
            empty = np.asarray(self.renderer.render_to_image()).astype(np.float64)
        for k, (metal, ghost) in enumerate([(False, False), (True, False), (False, True)]):
            self._add(f'prewarm{k}', V + [2.0 * (k - 1), 0.0, 0.0], N, F, (0.8, 0.1, 0.1), metal, ghost)
        h, w = self.height * self.ss, self.width * self.ss
        spots = [(slice(h // 2 - h // 30, h // 2 + h // 30), slice(c - w // 60, c + w // 60))
                 for c in (w // 2 - int(w * 0.175), w // 2, w // 2 + int(w * 0.175))]
        for frame in range(max_frames):
            self.renderer.setup_camera(FOV, [0, 0, 0.5], [0, -12, 4.5], [0, 0, 1])
            img = np.asarray(self.renderer.render_to_image()).astype(np.float64)
            if all(np.abs(img[r, c] - empty[r, c]).max(axis=-1).mean() > 8 for r, c in spots):
                break
        else:
            raise RuntimeError(f'the renderer drew no objects in {max_frames} frames')
        scene.clear_geometry()
        self.prewarm_frames = frame + 1

    def _floor(self):
        o3d, rendering, L = self.o3d, self.rendering, self.look
        half, reps = FLOOR_SIZE / 2, FLOOR_SIZE / (2 * TILE)          # the texture holds 2 x 2 tiles and repeats
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        mesh.triangle_uvs = o3d.utility.Vector2dVector([[0, 0], [reps, 0], [reps, reps], [0, 0], [reps, reps], [0, reps]])
        mesh.compute_vertex_normals()
        a, b = (np.asarray(c) for c in L['checker'])
        n = 64                                                          # texels per tile
        check = (np.arange(2 * n)[:, None] // n + np.arange(2 * n)[None, :] // n) % 2
        tex = np.where(check[..., None] > 0, b, a)
        srgb = np.where(tex <= 0.0031308, tex * 12.92, 1.055 * tex ** (1 / 2.4) - 0.055)   # albedo images are sRGB
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        mat.albedo_img = o3d.geometry.Image(np.ascontiguousarray((srgb * 255).round().astype(np.uint8)))
        mat.base_roughness = L['floor_roughness']
        mat.base_reflectance = L['floor_reflectance']
        return mesh, mat

    def _add(self, name, V, N, F, color, metal=False, ghost=False, alpha=None):
        """alpha: draw the mesh in `color` with this opacity and without shadows (style 'points')."""
        o3d, rendering, L = self.o3d, self.rendering, self.look
        # int32 triangles: with int64 ones and given vertex normals the engine draws nothing but the shadow
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.ascontiguousarray(V, np.float64)),
                                         o3d.utility.Vector3iVector(np.ascontiguousarray(F, np.int32)))
        mesh.vertex_normals = o3d.utility.Vector3dVector(np.ascontiguousarray(N, np.float64))
        mesh.compute_triangle_normals()
        mat = rendering.MaterialRecord()
        if ghost:
            mat.shader = 'defaultLitTransparency'
            mat.base_color = list(GHOST)
            mat.base_roughness = 0.6
        elif alpha is not None:                     # matte, little specular: the colour shows through
            mat.shader = 'defaultLitTransparency'
            mat.base_color = [*color, float(alpha)]
            mat.base_roughness, mat.base_reflectance = 0.8, 0.1
        else:
            mat.shader = 'defaultLit'
            mat.base_color = [*color, 1.0]
            if metal:
                mat.base_metallic, mat.base_roughness = 1.0, L['metal_roughness']
            else:
                mat.base_metallic, mat.base_roughness = 0.0, L['roughness']
                mat.base_reflectance = L['reflectance']
                mat.base_clearcoat, mat.base_clearcoat_roughness = L['clearcoat'], L['clearcoat_roughness']
        self.renderer.scene.add_geometry(name, mesh, mat)
        if ghost or alpha is not None:                  # the ground truth and see-through meshes cast no shadow
            self.renderer.scene.scene.geometry_shadows(name, False, False)

    def render(self, out_path, meshes, sizes, metal, poses, fps, ghost=None, times=None, points=None,
               point_radius=0.025, colours=None, view=None, style='solid', see_through=SEE_THROUGH, added=None):
        """meshes[i] = (vertices, normals, faces) in the object's frame, unscaled; sizes[i] its scale; metal[i] bool;
        poses[i] (K, 4, 4) its key poses; ghost: key poses drawn translucent (the ground truth), or None. With
        `times` (K,), the video time (s) of each key pose, the video samples the motion at `fps`, interpolating
        between key poses (interpolate_poses); without, every key pose is one frame. points[i] (N_i, 3) or None:
        surface points of object i in its frame (unscaled), drawn as spheres of `point_radius` (m) that move with the
        object (the observed points of partial inputs); added[i] (M_i, 3) or None: points a completion network added
        to points[i] (data.partial_view.completion), drawn with ADDED_RADIUS times that radius. style 'solid': opaque
        objects with light points on them; 'points': the points opaque in the object's colour (darkened by
        POINT_SHADE, added points lightened by ADDED_TINT), the object see-through around them (opacity
        `see_through`, its colour lightened by GLASS_TINT). colours[i]: sRGB hex colour of object i (default: PALETTE
        by index). view: direction from the objects to the camera (default VIEW); the key light turns with it about
        the vertical, so it keeps lighting the objects from above, left of the camera."""
        if style not in ('solid', 'points'):
            raise ValueError(f"style must be 'solid' or 'points', got {style!r}")
        view = VIEW if view is None else tuple(float(v) for v in view)
        turn = np.arctan2(view[1], view[0]) - np.arctan2(VIEW[1], VIEW[0])
        c, s = np.cos(turn), np.sin(turn)
        kx, ky, kz = KEY_DIRECTION
        self.renderer.scene.scene.set_sun_light([c * kx - s * ky, s * kx + c * ky, kz], list(self.look['key']),
                                                self.look['key_intensity'])
        if times is not None:
            t = np.arange(int(np.floor(float(times[-1] - times[0]) * fps + 1e-6)) + 1) / fps + float(times[0])
            poses = [interpolate_poses(P, times, t) for P in poses]
            ghost = None if ghost is None else [interpolate_poses(P, times, t) for P in ghost]
        scene = self.renderer.scene
        scene.clear_geometry()
        scene.add_geometry('floor', *self.floor)
        layers = [('obj', poses)] + ([('gt', ghost)] if ghost is not None else [])
        marked = [i for i in range(len(meshes)) if points is not None and points[i] is not None and len(points[i])]
        rgb = [linear_rgb(colours[i] if colours is not None else PALETTE[i % len(PALETTE)]) for i in range(len(meshes))]
        for tag, _ in layers:
            for i, (V, N, F) in enumerate(meshes):
                glass = style == 'points' and tag == 'obj' and i in marked
                colour = (1 - GLASS_TINT) * rgb[i] + GLASS_TINT if glass else rgb[i]
                self._add(f'{tag}{i}', V * float(sizes[i]), N, F, colour, bool(metal[i]), ghost=tag == 'gt',
                          alpha=float(see_through) if glass else None)
        for i in marked:
            self._add(f'pts{i}', *point_spheres(np.asarray(points[i], np.float64) * float(sizes[i]), point_radius),
                      POINT_SHADE * rgb[i] if style == 'points' else POINT)
        filled = [i for i in marked if added is not None and added[i] is not None and len(added[i])]
        for i in filled:
            self._add(f'add{i}', *point_spheres(np.asarray(added[i], np.float64) * float(sizes[i]),
                                                ADDED_RADIUS * point_radius),
                      (1 - ADDED_TINT) * rgb[i] + ADDED_TINT if style == 'points' else POINT)
        radii = [float(np.linalg.norm(V, axis=1).max()) * float(sizes[i]) for i, (V, _, _) in enumerate(meshes)]
        target, eye = frame_camera(poses, radii, self.width / self.height, view=view)
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{self.width}x{self.height}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '18',
               '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path)]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            # the engine shows newly added geometry only after a couple of frames: render the first frame twice ahead
            for tag, seq in layers:
                for i in range(len(meshes)):
                    scene.set_geometry_transform(f'{tag}{i}', seq[i][0])
            for i in marked:
                scene.set_geometry_transform(f'pts{i}', poses[i][0])
            for i in filled:
                scene.set_geometry_transform(f'add{i}', poses[i][0])
            self.renderer.setup_camera(FOV, target, eye, [0, 0, 1])
            for _ in range(2):
                self.renderer.render_to_image()
            for t in range(len(poses[0])):
                for tag, seq in layers:
                    for i in range(len(meshes)):
                        scene.set_geometry_transform(f'{tag}{i}', seq[i][t])
                for i in marked:
                    scene.set_geometry_transform(f'pts{i}', poses[i][t])
                for i in filled:
                    scene.set_geometry_transform(f'add{i}', poses[i][t])
                # after the objects are placed: the camera's clipping planes follow the scene's bounds
                self.renderer.setup_camera(FOV, target, eye, [0, 0, 1])
                img = np.asarray(self.renderer.render_to_image())
                if self.ss > 1:
                    img = img.reshape(self.height, self.ss, self.width, self.ss, 3).mean(axis=(1, 3))
                enc.stdin.write(np.ascontiguousarray(img.round().astype(np.uint8)).tobytes())
        finally:
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError(f'ffmpeg failed to write {out_path}')


_RENDERER = None


def get_renderer(width=1280, height=720, supersample=2, floor='light'):
    """One renderer per process (the Filament engine is created once)."""
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = RolloutRenderer(width, height, supersample, floor)
    elif (_RENDERER.width, _RENDERER.height, _RENDERER.ss, _RENDERER.floor_tone) != (width, height, supersample, floor):
        raise ValueError(f'the renderer of this process is {_RENDERER.width}x{_RENDERER.height} (supersample '
                         f'{_RENDERER.ss}, {_RENDERER.floor_tone} floor), requested {width}x{height} (supersample '
                         f'{supersample}, {floor} floor)')
    return _RENDERER
