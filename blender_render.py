import bpy
import random
import math
from mathutils import Vector, Euler
from pathlib import Path
import time

DATASET_ROOT = Path("D:/oakeshott_rendered/Type_XIV")
#DATASET_ROOT = Path("D:/temp_set")
TYPE_NAME = "XIV"

SPLITS = [("train", 150), ("val", 15), ("test", 15)]
#SPLITS = [("train", 15)]


CAMERA_NAME = "Camera"
KEY_LIGHT_NAME = "Key Light"
FILL_LIGHT_NAME = "Fill Light"

SWORD_MODELS = [
    bpy.data.objects["Sketchfab_model"],
    bpy.data.objects["Sketchfab_model.001"],
    bpy.data.objects["Sketchfab_model.002"],
   # bpy.data.objects["Sketchfab_model.003"],
]

# Background colors: neutral grays, off-whites, light beiges, occasionally darker
# Each entry is (R, G, B) in linear space (0-1)
BACKGROUND_COLORS = [
    (0.90, 0.90, 0.90),  # near white
    (0.80, 0.80, 0.80),  # light gray
    (0.70, 0.70, 0.70),  # mid gray
    (0.60, 0.60, 0.60),  # darker gray
    (0.88, 0.85, 0.80),  # warm off-white
    (0.85, 0.83, 0.78),  # light beige
    (0.82, 0.80, 0.75),  # beige
    (0.50, 0.50, 0.50),  # museum-style mid tone
    (0.95, 0.95, 0.95),  # almost white seamless
    (0.40, 0.38, 0.35),  # occasionally darker warm tone
]

camera = bpy.data.objects[CAMERA_NAME]
key    = bpy.data.objects[KEY_LIGHT_NAME]
fill   = bpy.data.objects[FILL_LIGHT_NAME]

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 384
scene.render.resolution_y = 384
scene.render.resolution_percentage = 100
scene.eevee.taa_render_samples = 64  # bump from 32 for cleaner edges



def look_at(obj, target: Vector):
    direction = target - obj.location
    rot_quat  = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


def set_hierarchy_render_visibility(root_obj, visible: bool):
    for o in [root_obj] + list(root_obj.children_recursive):
        o.hide_render   = not visible
        o.hide_viewport = not visible


def activate_only(root_obj):
    for m in SWORD_MODELS:
        set_hierarchy_render_visibility(m, False)
    set_hierarchy_render_visibility(root_obj, True)


def world_bbox_center_and_radius(objs):
    pts = []
    for obj in objs:
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    min_v = Vector((min(xs), min(ys), min(zs)))
    max_v = Vector((max(xs), max(ys), max(zs)))
    return (min_v + max_v) / 2.0, (max_v - min_v).length / 2.0


def get_model_objects_for_bbox(root_obj):
    objs = [o for o in [root_obj] + list(root_obj.children_recursive)
            if o.type == "MESH"]
    return objs if objs else [root_obj]



def randomize_camera(sword_center: Vector, sword_radius: float):
    """
    Expanded from original:
      - yaw  ±12° → ±30°   (more 3/4 views)
      - pitch  8-18° → -10° to +40°  (below-center & overhead shots)
      - ~10 % chance of a negative-pitch shot (slightly below the sword)
    """
    yaw   = random.uniform(-math.radians(30), math.radians(30))

    if random.random() < 0.10:               # 10 % below-center
        pitch = random.uniform(math.radians(-10), math.radians(0))
    else:
        pitch = random.uniform(math.radians(5), math.radians(40))

    dist = (3.2 * sword_radius) * random.uniform(1.00, 1.10)

    x = dist * math.cos(yaw) * math.cos(pitch)
    y = dist * math.sin(yaw) * math.cos(pitch)
    z = dist * math.sin(pitch)

    camera.location = sword_center + Vector((x, y, z))
    look_at(camera, sword_center)


def randomize_lights():

    roll = random.random()

    if roll < 0.60:          # standard
        key.data.energy  = 1100 * random.uniform(0.80, 1.20)
        fill.data.energy =  550 * random.uniform(0.80, 1.20)

    elif roll < 0.85:        # diffuse / museum overcast
        base = random.uniform(700, 1000)
        key.data.energy  = base * random.uniform(0.95, 1.05)
        fill.data.energy = base * random.uniform(0.85, 1.00)   # nearly equal

    else:                    # dramatic
        key.data.energy  = 1400 * random.uniform(0.90, 1.10)
        fill.data.energy =  150 * random.uniform(0.80, 1.20)

    # Orbit the key light randomly around the sword so shadow direction varies
    key_orbit_angle = random.uniform(0, math.pi * 2)
    key_dist        = random.uniform(3.0, 5.0)
    key.location    = Vector((
        key_dist * math.cos(key_orbit_angle),
        key_dist * math.sin(key_orbit_angle),
        random.uniform(2.5, 5.0),
    ))

    fill_orbit_angle = key_orbit_angle + math.pi + random.uniform(-0.5, 0.5)
    fill_dist        = random.uniform(3.5, 5.5)
    fill.location    = Vector((
        fill_dist * math.cos(fill_orbit_angle),
        fill_dist * math.sin(fill_orbit_angle),
        random.uniform(1.5, 4.0),
    ))


def randomize_background():
    color = random.choice(BACKGROUND_COLORS)
    world = bpy.context.scene.world
    if world and world.use_nodes:
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = (*color, 1.0)
            bg_node.inputs["Strength"].default_value = 1.0


def randomize_sword_roll(root_obj):

    roll = random.choice([
        0,                                           # upright
        0,                                           # upright again
        random.uniform(-math.radians(15), math.radians(15)),   # slight tilt
        random.uniform(math.radians(40), math.radians(50)),    # ~45°
        random.uniform(-math.radians(50), -math.radians(40)),  # ~-45°
        math.radians(90),                            # horizontal
        math.radians(-90),                           # horizontal other way
    ])
    root_obj.rotation_euler = Euler((
        root_obj.rotation_euler.x,
        root_obj.rotation_euler.y,
        root_obj.rotation_euler.z + roll,
    ), 'XYZ')
    return roll

start_time = time.time()

for split_name, count in SPLITS:
    output_dir = DATASET_ROOT / split_name / TYPE_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        # Choose and activate model
        model = random.choice(SWORD_MODELS)
        activate_only(model)

        # Apply in-plane roll to sword before computing bbox
        #    so the bbox (and therefore camera distance) stays correct
        applied_roll = randomize_sword_roll(model)

        # Compute bbox for positioned model
        bbox_objs = get_model_objects_for_bbox(model)
        sword_center, sword_radius = world_bbox_center_and_radius(bbox_objs)

        # Camera, lights, background
        randomize_camera(sword_center, sword_radius)
        randomize_lights()
        randomize_background()

        # Render
        scene.render.filepath = str(output_dir / f"{i:05d}.png")
        bpy.ops.render.render(write_still=True)

        # Reset sword roll so next iteration starts clean
        model.rotation_euler = Euler((
            model.rotation_euler.x,
            model.rotation_euler.y,
            model.rotation_euler.z - applied_roll,
        ), 'XYZ')

        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{split_name}] {i}/{count} | elapsed {elapsed:.1f}s")

# Restore visibility
for m in SWORD_MODELS:
    set_hierarchy_render_visibility(m, True)

print("Rendering complete.")
