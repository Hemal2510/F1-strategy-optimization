class TyreModel:
    def __init__(self, data):
    
        self.degradation_curves = data["tyre_loss"]


    def degradation(self,compound, age):
        curve = self.degradation_curves[compound]
        max_age = max(curve.keys())
        clipped_age = min(age, max_age)
        clipped_age = max(1, clipped_age)
        return curve[clipped_age]