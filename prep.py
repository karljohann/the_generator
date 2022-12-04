import csv
import glob
import os
from pathlib import Path

import numpy as np

import pretty_midi as pm


INPUT_FILES_PATH = "/Users/karljohann/Downloads/archive/Jazz Midi"
STORAGE_PATH = "/Users/karljohann/Downloads/the_generator/csv/"


def output_file(filename, filetype, appendix=""):
    return f"{STORAGE_PATH}{Path(filename).stem}{appendix}.{filetype}"


def read_csv(filename):
    path = f"{STORAGE_PATH}/{filename}.csv"
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def write_csv(rows, filename, appendix=""):
    out = f"{','.join(rows[0].keys())}\n"
    for row in rows:
        out += f"{str(','.join([str(i) for i in row.values()]))}\n"

    with open(output_file(filename, 'csv', appendix), 'wt') as f:
        f.write(out)

def processFile(f):
    mid = pm.PrettyMIDI(f)

    # time (ms) since last note (notes in chords all have same start time)
    def getTimedelta(note, last_timedelta, last_note):
        timedelta = (note.start - last_note.start)
        # if timedelta == 0: # if first note(s) not at start of song (timedelta is initialised as 0)
        #     return 0.0 # FIXME: dont have to, it should just be 0 (last_note is initialised as the first)

        # if last_note.start == note.start: # same start time as last: chord (or first note)
        #     return last_timedelta

        if last_note.start == note.start: # same start time as last: chord (or first note)
        # if last_note.start == note.start or last_note.end == note.end: # same start time as last: chord (or first note)
            return last_timedelta

        # if timedelta < 0: # TODO: Can probably combine this if with the one above
        #     if last_note.end == note.end: # well, then you've got a chord, I guess
        #         return last_timedelta
        #     else:
        #         return 0.0 # FIXME: not sure what's going on

        return np.clip(timedelta, 0.0, 999)


    tracks = []
    for i, instrument in enumerate(mid.instruments): # TODO: remove enumeration (only used for filename)
        if not instrument.is_drum: # skip instruments that are not melodic
            last_note = instrument.notes[0]
            timedelta = 0
            notes = []

            instr_notes = instrument.notes
            instr_notes = sorted(instr_notes, key=lambda k: k.start)
            for note in instr_notes:
                timedelta = getTimedelta(note, timedelta, last_note)
                last_note = note

                notes.append({
                    'note': pm.note_number_to_name(note.pitch),
                    'time': round(note.start, 3),
                    'end': round(note.end, 3),
                    'velocity': note.velocity,
                    'timedelta': round(timedelta, 3),
                    'note_length': note.get_duration(),

                    'note_int': note.pitch,
                    'tick': mid.time_to_tick(note.start),
                    'tickdelta': mid.time_to_tick(note.start - timedelta),
                    'note_length_tick': mid.time_to_tick(note.get_duration()),
                })

        write_csv(notes, f, f"{str(i)}_{instrument.name}")

        tracks.append(notes)

    # time, vel, pitch, prog = zip(*events)
    # torch.save(dict(
    #     time = torch.DoubleTensor(columns['timedelta']),
    #     pitch = torch.LongTensor(columns['note']),
    #     velocity = torch.LongTensor(columns['velocity']),
    #     # length = torch.LongTensor(columns['length']),
    # ), output_file(f, 'pkl'))

    return tracks


if __name__ == "__main__":
    def get_files(dir = None):
        files = []
        for file_path in list(glob.glob(dir)):
            _, ext = os.path.splitext(file_path)
            if ext == '.mid':
                files.append(file_path)
            if os.path.isdir(file_path):
                files += get_files(file_path + '/*.mid')

        return files

    files = get_files(INPUT_FILES_PATH + "/**") # all files and subfolders
    asd = 0
    for f in files:
        print(f)
        (processFile(f))
        # pprint(processFile(f))
        if asd == 1:
            break
        asd +=1
