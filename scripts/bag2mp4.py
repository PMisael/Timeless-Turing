import pyrealsense2 as rs, cv2, numpy as np, argparse, os
from pathlib import Path

# ---------- argumentos ----------
ap = argparse.ArgumentParser()
ap.add_argument("bag", help="captura .bag")
ap.add_argument("-o", "--out", help="fichero de salida")
ap.add_argument("--min", type=float, default=0.15, help="mín. distancia [m]")
ap.add_argument("--max", type=float, default=1.00, help="máx. distancia [m]")
args = ap.parse_args()

if args.out is None:
    bag_path = Path(args.bag)
    # Mismo nombre, extensión .mp4
    args.out = str(bag_path.with_suffix(".mp4"))
#
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_device_from_file(args.bag, repeat_playback=False)
profile   = pipe.start(cfg)

align     = rs.align(rs.stream.color)     # prof. → geometría color
depth_sc  = profile.get_device().first_depth_sensor().get_depth_scale()

fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
writer    = None                          # se crea cuando sepamos W×H
kernel    = np.ones((5,5), np.uint8)      # cierre morfológico

formato=True

print("Procesando …")
try:
    while True:
        fs = pipe.wait_for_frames()
        if not fs:        # EOF
            break
        fs = align.process(fs)
        d  = fs.get_depth_frame()
        c  = fs.get_color_frame()
        if formato:
            fmt = c.get_profile().format()
            print("Formato color:", fmt)
            formato=False  
        if not d or not c:
            continue

        # --- máscara por profundidad ---
        depth = np.asanyarray(d.get_data(), dtype=np.float32) * depth_sc
        mask  =  (depth > args.min) & (depth < args.max)
        mask  = (mask + 1) / 2 
        mask  = cv2.morphologyEx(mask.astype(np.uint8)*255,
                                 cv2.MORPH_CLOSE, kernel, iterations=2)
        
        #breakpoint()
        
        # Buffer original (RGB)
        color_rgb = np.asanyarray(c.get_data()) # BGR uint8

        # Convertir a BGR para OpenCV / VideoWriter
        color = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        nobg  = cv2.bitwise_and(color, color, mask=mask)

        # inicializa VideoWriter una vez conocemos tamaño ---
        if writer is None:
            h, w = nobg.shape[:2]
            fps = c.get_profile().fps() 
            writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        writer.write(nobg)
except Exception as e:
    pass      # se alcanza fin natural del .bag
finally:
    pipe.stop()
print("Vídeo generado →", os.path.abspath(args.out))
