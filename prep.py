import glob
import os
from pathlib import Path

import numpy as np

import pretty_midi as pm


INPUT_FILES_PATH = "/Users/karljohann/Downloads/the_generator/bach/musicnet_midis/musicnet_midis/Bach/"
STORAGE_PATH = "/Users/karljohann/dev/HR/the_generator/data/csv/bach/"


def get_files(dir = None):
    files = []
    for file_path in list(glob.glob(dir)):
        _, ext = os.path.splitext(file_path)
        if ext == '.mid':
            files.append(file_path)
        if os.path.isdir(file_path):
            files += get_files(file_path + '/*.mid')

    return files

def output_file(filename, filetype, appendix=""):
    return f"{STORAGE_PATH}{Path(filename).stem}{appendix}.{filetype}"

def write_csv(rows, filename, appendix=""):
    out = f"{','.join(rows[0].keys())}\n"
    for row in rows:
        out += f"{str(','.join([str(i) for i in row.values()]))}\n"

    with open(output_file(filename, 'csv', appendix), 'wt') as f:
        f.write(out)

def processFile(f):
    try:
        mid = pm.PrettyMIDI(f)
    # except OSError as e:
    # except ValueError as e:
    except Exception as e:
        print(f"File ({Path(f).name}) failed:", e)
        return None

    # time (ms) since last note
    def getTimedelta(note, last_timedelta, last_note):
        if last_note.start == note.start: # same start time as last: chord (or first note) # FIXME: Set chord as 0.0, otherwise it's predicting next note in chord
            return 0.0 # FIXME: Doesn't this happen anyway if last_note == note?
            # return last_timedelta

        return np.clip((note.start - last_note.start), 0.0, 999.9)

    for i, instrument in enumerate(mid.instruments): # TODO: remove enumeration (only used for filename)
        notes = None
        # if not instrument.is_drum and instrument.program in list(range(32, 40)): # skip instruments that are not melodic
        if not instrument.is_drum: # skip instruments that are not melodic
            last_note = instrument.notes[0]
            timedelta = 0
            notes = []

            instr_notes = instrument.notes
            instr_notes = sorted(instr_notes, key=lambda k: k.start)
            for note in instr_notes:
                # timedelta = getTimedelta(note, timedelta, last_note)
                # timedelta = np.clip((note.start - last_note.start), 0.0, 99.9)
                timedelta = (note.start - last_note.start)
                last_note = note


                notes.append({
                    'instrument': instrument.program,
                    'instrument_name': instrument.name,

                    'note': pm.note_number_to_name(note.pitch),
                    'time': round(note.start, 3),
                    'end': round(note.end, 3),
                    'velocity': note.velocity,
                    'timedelta': round(timedelta, 3),
                    'note_length': round(note.get_duration(), 3),

                    'note_int': note.pitch,
                    'tick': mid.time_to_tick(note.start),
                    'tickdelta': mid.time_to_tick(note.start - timedelta),
                    'note_length_tick': mid.time_to_tick(note.get_duration()),
                })

        if notes:
            write_csv(notes, f, f"{str(i)}")


if __name__ == "__main__":
    files = get_files(INPUT_FILES_PATH + "/**") # all files and subfolders
    fsize = len(files)
    for i, f in enumerate(files):
        print(f"{(i + 1)}/{fsize}: {Path(f).name}")
        processFile(f)
