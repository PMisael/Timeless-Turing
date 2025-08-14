## ⚙️ Configuración de Visual Studio Code

El repositorio incluye una carpeta `.vscode` con la configuración de entorno y notebooks:

```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "jupyter.envFile": "${workspaceFolder}/.env",
    "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```
---
Esto asegura que:
  - Los notebooks de Jupyter usen las variables definidas en .env.
  - El entorno de Python sea consistente tanto en scripts como en notebooks.
  - La raíz para notebooks sea la carpeta del proyecto.
