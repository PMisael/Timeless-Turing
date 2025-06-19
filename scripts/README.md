# Parámetros clave:

--video.bag

--distancia mínima para el recorte (metros)

--distancia máxima para el recorte (metros)

---
## Uso rápido
### Nombre por defecto (cambia extension .bag a .mp4)
```bash
python bag2mp4.py video.bag \
                  --min 0.5 \
                  --max 1.0 \
```

### Nombre y ruta personalizada del video mp4 resultante

```bash
python bag2mp4.py video.bag \
                  ruta/del/video_sin_fondo.mp4 \
                  --min 0.5 \
                  --max 1.0 \
```

La salida muestra la ruta del video sin fondo creado en formato mp4 el cual tiene el mismo nombre que el archivo.bag procesado.
