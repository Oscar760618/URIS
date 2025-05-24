# This file should be run at python environment 3.8 because an old version of tensorflow is needed
# ------------------------------------------------------------------------------------------------

'''
code for processing the midi data, convert them into lists saved as numpy arrays
'''
import os
import torch
import numpy as np
from tqdm import tqdm
import pretty_midi
import pypianoroll
import music21
import note_seq
from MidiPerformanceEncoder import MidiPerformanceEncoder
import traceback
from utils2 import encode_midi

midi_data_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/Mus_dataset'
save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/All_Data/'

PR_TIME_STEPS = 64
NUM_VELOCITY_BINS = 64
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108
MIN_NOTE_DENSITY = 0
MAX_NOTE_DENSITY = 13
MIN_TEMPO = 57
MAX_TEMPO = 258
MIN_VELOCITY = 0
MAX_VELOCITY = 126

del_lst = []

def magenta_encode_midi(midi_filename, is_eos=False):
    '''
    Encode aMIDI file into a sequence of data.
    '''
    mpe = MidiPerformanceEncoder(
            steps_per_second=STEPS_PER_SECOND,
            num_velocity_bins=NUM_VELOCITY_BINS,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
            add_eos=is_eos)
    ns = note_seq.midi_file_to_sequence_proto(midi_filename)
    return mpe.encode_note_sequence(ns)

def magenta_decode_midi(notes, is_eos=False, file_path=None):
    """
    Decode a sequence of notes into a MIDI file.
    """
    mpe = MidiPerformanceEncoder(
        steps_per_second=STEPS_PER_SECOND,
        num_velocity_bins=NUM_VELOCITY_BINS,
        min_pitch=MIN_PITCH,
        max_pitch=MAX_PITCH,
        add_eos=is_eos
    )
    midi_file_path = mpe.decode(notes, strip_extraneous=False, file_path=file_path)
    return midi_file_path

def slice_midi(pm, beats, start_idx, end_idx):
    '''
    Slice given pretty_midi object into number of beat segments.
    '''
    new_pm = pretty_midi.PrettyMIDI()
    new_inst = pretty_midi.Instrument(program=pm.instruments[0].program,
                                      is_drum=pm.instruments[0].is_drum,
                                      name=pm.instruments[0].name)
    start, end = beats[start_idx], beats[end_idx]
    for i in range(len(pm.instruments)):
        for note in pm.instruments[i].notes:
            velocity, pitch = note.velocity, note.pitch
            if note.start > end or note.start < start:
                continue
            else:
                s = note.start - start
                if note.end > end:
                    e = end - start
                else:
                    e = note.end - start
            new_note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=s, end=e)
            new_inst.notes.append(new_note)

        for ctrl in pm.instruments[i].control_changes:
            if ctrl.time >= start and ctrl.time < end:
                new_ctrl = pretty_midi.ControlChange(
                    number=ctrl.number, value=ctrl.value, time=ctrl.time - start)
                new_inst.control_changes.append(new_ctrl)

    new_pm.instruments.append(new_inst)
    new_pm.write('tmp.mid')
    return new_pm

def get_harmony_vector(fname, is_one_hot=False):
    '''
    Obtain estimated key for a given music segment with music21 library.
    '''
    CHORD_DICT = {
    "C-": 11, "C": 0, "C#": 1, "D-": 1, "D": 2, "D#": 3, "E-": 3, "E": 4, "E#": 5,
    "F-": 4, "F": 5, "F#": 6, "G-": 6, "G": 7, "G#": 8, "A-": 8, "A": 9, "A#": 10, 
    "B-": 10, "B": 11, "B#": 0
    }

    try:
        score = music21.converter.parse(fname)
        key = score.analyze('key')
        res = np.zeros(24,)
        name, mode = key.tonic.name, key.mode
        idx = CHORD_DICT[name] + 12 if mode == "minor" else CHORD_DICT[name]

        if not is_one_hot: 
            res[idx] = key.correlationCoefficient
            for i, x in enumerate(key.alternateInterpretations):
                name, mode = x.tonic.name, x.mode
                idx = CHORD_DICT[name] + 12 if mode == "minor" else CHORD_DICT[name]
                res[idx] = x.correlationCoefficient

            res[res < 0.1] = 0
        else:
            if idx:
                res[idx] = 1

        return res

    except Exception as e:
        print(e, "harmony vector")
        return None

def get_music_attributes(pr, beat=24):
    '''
    Get musical attributes including rhythm density, note_density, chroma and velocity
    for a given piano roll segment.
    '''
    events, pitch_lst, pr, rhythm = encode_midi(pr, beat=beat, is_pr=True)

    # get note density
    note_density = np.array([len(k) for k in pitch_lst])

    # get chroma
    chroma = np.zeros((pr.shape[0], 12))
    for note in range(12):
        chroma[:, note] = np.sum(pr[:, note::12], axis=1)

    return events, rhythm, note_density, chroma

def process_data(name, beat_res=4, num_of_beats=4, max_tokens=100):
    '''
    Utility function for each data function to extract required data. Referenced in get_classic_piano function.
    '''
    data_lst = []
    rhythm_lst = []
    note_density_lst = []
    chroma_lst = []
    
    track = pypianoroll.read(name).tracks

    if len(track) > 0:
        try:
            pm = pretty_midi.PrettyMIDI(name)
            beats = pm.get_beats()
        except Exception as e:
            print(e)

        pr = track[0].pianoroll

        for j in range(0, len(pr), beat_res * num_of_beats):
            start_idx = j
            end_idx = j + beat_res * num_of_beats

            if end_idx // beat_res >= len(beats):
                end_idx = (len(beats) - 1) * beat_res
                if start_idx >= end_idx:
                    break
            
            new_pr = pr[start_idx : end_idx]
            new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
            new_pm.write("tmp.mid")

            if len(new_pm.instruments[0].notes) > 0:

                # get musical attributes
                _, rhythm, note_density, chroma = get_music_attributes(new_pr, beat=beat_res)

                events = magenta_encode_midi("tmp.mid")
                events.append(1)

                if len(events) <= max_tokens:   
                    chroma = get_harmony_vector("tmp.mid", is_one_hot=True)       
                    data_lst.extend(events)
                    rhythm_lst.extend(rhythm)
                    note_density_lst.extend(note_density)
                    chroma_lst.extend(chroma)
        
    return torch.Tensor(data_lst), rhythm_lst, note_density_lst, chroma_lst

def get_classic_piano(midi_data_path, data_type="long"):
    '''
    Saving the preprocessed midi data into numpy arrays.
    '''
    files = os.listdir(midi_data_path)
    files.sort()
    labelled_midi = [os.path.join(midi_data_path, k) for k in files]

    print("Dataset length:", len(labelled_midi))
    keylst = labelled_midi

    data_lst = []
    rhythm_lst = []
    note_density_lst = []
    chroma_lst = []
    
    for i, name in tqdm(enumerate(keylst), total=len(keylst)):
        try:
            print(f"processing {name}")
            if data_type == "short":
                beat_res, num_of_beats, max_tokens = 4, 4, 100
            elif data_type == "long":
                beat_res, num_of_beats, max_tokens = 4, 16, 1000
                

            cur_data_lst, cur_rhythm_lst, cur_note_lst, cur_chroma_lst = process_data(name,
                                                                                        beat_res=beat_res, 
                                                                                        num_of_beats=num_of_beats, 
                                                                                        max_tokens=max_tokens)
            print(len(cur_data_lst), len(cur_rhythm_lst), len(cur_note_lst), len(cur_chroma_lst))
            data_lst.append(cur_data_lst)
            rhythm_lst.append(cur_rhythm_lst)
            note_density_lst.append(cur_note_lst)
            chroma_lst.append(cur_chroma_lst)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            print(traceback.format_exc())
            del_lst.append(name)
        
        if os.path.exists("tmp.mid"):
            os.remove("tmp.mid")

    print(len(data_lst), len(rhythm_lst), len(note_density_lst), len(chroma_lst))
    data_lst = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)
    rhythm_lst = np.array(rhythm_lst)
    note_density_lst = np.array(note_density_lst)
    chroma_lst = np.array(chroma_lst)
    
    print(del_lst)
    print("Shapes for: Data, Rhythm Density, Note Density, Chroma")
    print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)
    np.save(save_path + "data.npy", data_lst)
    np.save(save_path + "rhythm.npy", rhythm_lst)
    np.save(save_path + "note_density.npy", note_density_lst)
    np.save(save_path + "chroma.npy", chroma_lst)

    return data_lst, rhythm_lst, note_density_lst, chroma_lst

if __name__ == '__main__':
    get_classic_piano(midi_data_path,data_type="long")
