# Timeless-Turing
# Estimación del Neurodesarrollo Neonatal ― Prototipo con Simulación de Posturas

## Descripción

Este proyecto implementa un **pipeline** el cual parte desde la captura con **Intel RealSense D455** hasta la detección automática de **posturas**— como paso previo para estimar el neurodesarrollo neonatal mediante análisis de movimientos generales (GM).

1. **Captura** → Vídeos RGB‑D en formato `.bag`.
2. **Pre‑procesamiento** → Eliminación de fondo y exportación a `.mp4`.
3. **Landmark Detection** → **MediaPipe Pose** para identificar articulaciones y rastrear extremidades.
4. **Clasificación** → Red neuronal densa para etiquetar cada postura (p. ej. "saludo", "brazos abiertos", "brazos cruzados", "posición fetal").

---
