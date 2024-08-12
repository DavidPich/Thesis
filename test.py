import librosa
import numpy as np

'''# MIDI note number
'''# MIDI note number
midi_note = 10  # This is middle C

# Convert MIDI note to frequency in Hz
frequency = librosa.midi_to_hz(midi_note)

#print(f"The frequency of MIDI note {midi_note} is {frequency} Hz")

librosa.midi_to_hz(60)


print(f"The frequency of MIDI note C5 is {librosa.note_to_hz('C5')} Hz")

print(f"The frequency of MIDI note C3 is {librosa.note_to_hz('C3')} Hz")


f0 = np.array([312.929324, 311.12698372, 307.55338552])

"""Round the given pitch values to the nearest MIDI note numbers"""
midi_note = librosa.hz_to_midi(f0)
print(midi_note)
print(np.round(midi_note))

rounded = np.subtract(np.round(midi_note), midi_note)
print(rounded)
rounded = np.divide(rounded, 2)
print(rounded)
rounded = np.add(rounded, midi_note)
print(rounded)

# Convert back to Hz.
print(librosa.midi_to_hz(rounded))
'''

import numpy as np

# Angenommen, arr ist Ihr Array
arr = np.array([1, np.nan, 4, np.nan])

# Zählen der NaN Werte
nan_count = np.count_nonzero(np.isnan(arr))

# Berechnen der Gesamtanzahl der Elemente
total_elements = arr.size

# Berechnen des Prozentsatzes der NaN Werte
nan_percentage = (nan_count / total_elements) * 100

print(f"Prozentsatz der NaN Werte: {nan_percentage}%")

# Identify the pitch range of the audio
highestPitch = 0
lowestPitch = 100000
under100 = 0
between100and300 = 0
between300and600 = 0
between600and1200 = 0
over12 = 0
lowInfo = 0

def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    global highestPitch
    global lowestPitch
    global over12
    global under100
    global between100and300
    global between300and600
    global between600and1200

    midi_note = np.round(librosa.hz_to_midi(f0))

    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan

    for p in f0:
        if p > highestPitch and p < 1200:
            highestPitch = p
            
        if p > 1200:
            over12 += 1
        
        if p < 100:
            under100 += 1
        
        if p > 100 and p < 300:
            between100and300 += 1

        if p > 300 and p < 600:
            between300and600 += 1

        if p > 600 and p < 1200:
            between600and1200 += 1
        
        if p < lowestPitch:
            lowestPitch = p

    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)
'''

