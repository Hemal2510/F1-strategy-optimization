class RewardCalculator:
    def compute(self, lap_delta, start_pos, end_pos, gap_leader, prev_gap_leader, pitted=False, pit_loss=0.0):
        # time delta compared to the leader (positive if closing the gap, negative if losing ground)
        gap_delta = prev_gap_leader - gap_leader

        # If the driver pitted, adjust the gap delta by the physical pit loss
        # so they aren't double-penalized for the pit stop duration
        if pitted:
            gap_delta += pit_loss

      
        reward = (start_pos - end_pos) * 1.0 + gap_delta * 0.1 - lap_delta * 0.1
        return reward
       
