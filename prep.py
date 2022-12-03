import csv
import glob
import os
import midiutil
import mido

from collections import defaultdict
from pprint import pprint

from pathlib import Path

import numpy as np
import librosa
import librosa. display
import pyaudio

import pretty_midi as pmidi

import torch


STORAGE_PATH = "/Users/karljohann/Downloads/the_generator/pkl/"

def minmax_velocity(amps, lower = 70, upper = 127):
    ''' normalizes amplitude of onset to midi velocity '''
    amps = np.array(amps)
    amin = amps.min()
    amax = amps.max()
    return (amps - amin) / (amax - amin) * (upper - lower) + lower

def output_file(filename, filetype, appendix=""):
    # return STORAGE_PATH + Path(filename).with_suffix(filetype).name # FIXME: Is this needed?
    return STORAGE_PATH + Path(filename).stem + appendix + "." + filetype
    # return f"{STORAGE_PATH}{filename}.{filetype}"

def write_midi(hits, bpm, file_name_no_extension):
    MyMIDI = midiutil.MIDIFile(4)
    MyMIDI.addTempo(0, 0, bpm)
    quarter_note = 60 / bpm

    for track in range(0, 4): # four hands, four tracks
        MyMIDI.addProgramChange(track, 1, 0, 115) # 116 Woodblock

    for note, velocity, onset_time, hand in zip(*hits):
        # print(0, 0, int(note), onset_time / quarter_note, 0, (velocity))
        # track = np.random.randint(2) # TODO: heuristic player guessage
        track = np.random.randint(0, 4)
        MyMIDI.addNote(track, 0, int(note), (onset_time / quarter_note), 0.01, int(velocity))

    with open(output_file(file_name_no_extension, 'mid'), "wb") as f:
        MyMIDI.writeFile(f)


def read_csv(filename):
    path = f"{STORAGE_PATH}/{filename}.csv"
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def write_csv(rows, filename, appendix=""):
    out = ""
    out += "timedelta,note,note_int,velocity,length\n"
    # for timedelta, note, velocity, length in zip(row['timedelta'], row['note'], row['velocity'], row['length']):
    for row in rows:
        out += f"{row['time']},{row['note']},{row['note_int']},{row['velocity']},{row['note_length']}\n"
        # out += f"{timedelta},{note},{velocity},{length}\n"

    with open(output_file(filename, 'csv', appendix), 'wt') as f:
        f.write(out)


def processFile(f):
    failed_files = []
    try:
        mid = mido.MidiFile(f)
    # except OSError as e:
    # except ValueError as e:
    except Exception as e:
        print(f"File ({Path(f).name}) failed:", e)
        failed_files.append(Path(f).name)
        return None
    print("Failed files count:", len(failed_files), "\n", failed_files)

    tracks = []
    columns = {
        'note': [],
        'timedelta': [],
        'velocity': [],
        'length': [],
    }

    def closeRow(note, timestamp, velocity, note_length):
        return {
            'note': librosa.midi_to_note(note),
            'note_int': note,
            'time': timestamp,
            'velocity': velocity,
            'note_length': note_length,
        }

    for i, track in enumerate(mid.tracks):
        last_seen_time = 0
        elapsed_time = 0
        notes = []
        wait_for_off = defaultdict()
        absolute_time = defaultdict()

        n = 0
        for msg in track:
            elapsed_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                # TODO: Assume for overwriting notes and other notes without note_off
                if msg.note in wait_for_off: # If note already in dict and has not been released
                    row = closeRow(msg.note, msg.time, msg.velocity, 7)
                    notes.append(row)
                    del wait_for_off[msg.note]
                    del absolute_time[msg.note]

                wait_for_off[msg.note] = closeRow(
                    note=msg.note,
                    timestamp=(msg.time if msg.time > 0 else last_seen_time), # TODO : Should we just leave chords as they are?
                    velocity=msg.velocity,
                    note_length=0,
                )
                absolute_time[msg.note] = elapsed_time # time note started
                last_seen_time = (msg.time if msg.time > 0 else last_seen_time) # in case there are two notes at the same time. But what about really short between?
            elif msg.type == 'note_off':
                if msg.note in wait_for_off: # To keep the timelength of each note we must assume it is the same note and take the cumulative time
                    wait_row = wait_for_off.pop(msg.note)
                    abs_time = absolute_time.pop(msg.note)
                    row = closeRow(
                        note=wait_row['note_int'],
                        timestamp=wait_row['time'],
                        velocity=wait_row['velocity'],
                        note_length=(elapsed_time - abs_time),
                    )
                    notes.append(row)

                    columns['note'].append(wait_row['note_int'])
                    columns['timedelta'].append(wait_row['time'])
                    columns['velocity'].append(wait_row['velocity'])
                    columns['length'].append(wait_row['note_length'])
                    # del wait_for_off[msg.note]

            write_csv(notes, f, str(i))

        tracks.append(notes)

        # write_csv(columns, f, str(i))

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

    files = get_files('/Users/karljohann/Downloads/archive/Jazz Midi/**') # all files and subfolders
    asd = 0
    for f in files:
        print(f)
        pprint(processFile(f))
        if asd == 1:
            break
        asd +=1
