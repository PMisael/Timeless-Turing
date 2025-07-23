
class Prueba:
    def __init__(self):
        print('Iniciando Prueba')

    def Prediccion(self, model, data):
        best_model=model
        x_new=data
        prediccion  = best_model.predict(x_new)
        return (prediccion)

