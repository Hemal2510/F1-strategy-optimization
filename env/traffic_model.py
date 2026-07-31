class TrafficModel:
    def __init__(self):
        pass
    

    def get_traffic_loss(self, gap_ahead: float) -> float:
        if gap_ahead >= 3:
            return 0.0
        elif gap_ahead >= 1:
            return 0.4
        else:
            return 0.5


   
