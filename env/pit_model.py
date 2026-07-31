class PitModel:
    def __init__(self, data):
        self.pit_loss = data["pit_loss"]

    def get_loss(self, safety_car=0):
        loss = self.pit_loss
        # Both Full SC and VSC compress the field, reducing the opportunity cost
        # of a pit stop. SC (2) and VSC (1) both halve the effective pit loss.
        if safety_car > 0:
            loss *= 0.5
        return loss