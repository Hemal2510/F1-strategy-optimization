import numpy as np


def get_action_mask(env):
    # Mask allowed actions based on race state and weather rules
    mask = np.ones(6, dtype=bool)

    current_lap = env.state.current_lap
    max_laps = env.max_laps
    tyre_age = env.state.tyre_age
    current_compound = env.state.tyre_compound
    track_wetness = env.state.track_wetness

    mask[0] = True

    # Don't pit in early laps unless wet
    if current_lap < 3 and track_wetness == 0:
        mask[1:] = False
        return mask

    # Avoid back-to-back pit stops
    if tyre_age <= 2:
        mask[1:] = False
        return mask

    # Don't pit on final lap
    if current_lap >= max_laps - 1:
        mask[1:] = False
        return mask

    # Dry track: block inter and wet
    if track_wetness == 0:
        mask[4] = False
        mask[5] = False

    # Intermediate weather
    if 0 < track_wetness < 1.5:
        mask[1] = False
        mask[2] = False
        mask[3] = False
        mask[5] = False

    # Heavy rain
    if track_wetness >= 1.5:
        mask[1] = False
        mask[2] = False
        mask[3] = False
        mask[4] = False
        mask[5] = True

    # Can't pit to same compound
    same_compound_action = current_compound + 1
    if 1 <= same_compound_action <= 5:
        mask[same_compound_action] = False

    mask[0] = True
    return mask