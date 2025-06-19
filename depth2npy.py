"""
Extrae todos los frames de profundidad de un .bag
y los guarda como depth_000000.npy, …  (float32, metros)
"""
import pyrealsense2 as rs
import numpy as np
import os
import argparse
import tqdm
#
parser = argparse.ArgumentParser()
parser.add_argument("bag") # archivo .bag de entrada
parser.add_argument("-o", "--out",   default="depth_npy") # dir de salida
parser.add_argument("--max", type=int, default=None,      help="max frames")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_device_from_file(args.bag, repeat_playback=False)
cfg.enable_stream(rs.stream.depth, 0, 0, rs.format.z16, 30)  # resolución nativa
profile = pipe.start(cfg)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print("Depth scale:", depth_scale, "m / unidad")

try:
    for i in tqdm.trange(int(1e9)):                    # bucle “infinito”
        if args.max and i >= args.max:
            break
        frames = pipe.wait_for_frames()
        if not frames:
            break                                      # EOS

        d = np.asanyarray(frames.get_depth_frame().get_data(),
                          dtype=np.uint16).astype(np.float32)
        np.save(f"{args.out}/depth_{i:06d}.npy", d * depth_scale)   # metros
except Exception as e:                                   # fin natural del .bag
    print(f'Error: {e}')
finally:
    pipe.stop()
