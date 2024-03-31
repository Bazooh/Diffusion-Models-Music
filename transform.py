import pandas as pd
from scamp import *

def create_array(song: str, precision: int = 2, number_max_of_notes_per_chord: int = 4):
    """transform a song to an array

    Args:
        song (str): file path
        precision (int, optional): the greater this number is the greater the precision will be. Defaults to 2.
    Return:
        A list of notes where each note is a tuple (note, duration)
    """
    file = pd.read_csv(song)
    
    # list of (note, duration) for each 1/2th time
    
    start_beat = file["start_beat"]
    notes = [0. for _ in range(2 * (precision * int(start_beat[len(start_beat) - 1]) + 1) * number_max_of_notes_per_chord)]
    current_number_of_notes_per_chord = [0 for _ in range(precision * int(start_beat[len(start_beat) - 1]) + 1)]
    
    for note in file.values:
        # print(int(note[4] * precision))
        # if (int(note[4] * precision) > 100) : return
        current_number_of_notes_in_this_chord = current_number_of_notes_per_chord[int(note[4] * precision)]
        if current_number_of_notes_in_this_chord < number_max_of_notes_per_chord:
            duration = 2 * note[5] / 8
            duration = duration if duration < 1. else 1.
            value = note[3] / 100 if note[3] < 100 else 1.
            
            notes[int(note[4] * precision)*number_max_of_notes_per_chord + current_number_of_notes_in_this_chord] = value
            notes[(precision * int(start_beat[len(start_beat) - 1]) + 1) * number_max_of_notes_per_chord + int(note[4] * precision)*number_max_of_notes_per_chord + current_number_of_notes_in_this_chord] = duration
            
            current_number_of_notes_per_chord[int(note[4] * precision)] += 1
    
    print(notes[:100])
    
    return notes, number_max_of_notes_per_chord

def play_song_with_array(song: list, number_max_of_notes_per_chord: int, volume = 1.):
    s = Session()
    instrument = s.new_part("piano")
    
    notes = song[:len(song) // 2]
    durations = song[len(song) // 2:]
    
    i = 1
    for note, duration in zip(notes, durations):
        if duration > 0.05:
            instrument.play_note(note * 100, volume, duration * 8, blocking=False)
        
        print(note, duration)
        
        if i % number_max_of_notes_per_chord == 0:
            wait(0.15)
            i = 0
        
        i += 1

if __name__ == "__main__":
    play_song_with_array(*create_array("musicnet/test_labels/2106.csv"))