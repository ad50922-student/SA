import pickle
import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from music21 import stream
from music21 import note
from music21 import tempo
from music21 import meter


OUTPUT_DIR = "output_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

KEYS = [
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B"
]

BPM_RANGES = [
    (50, 70),
    (71, 90),
    (91, 110),
    (111, 130),
    (131, 150)
]

BARS_OPTIONS = [
    4,
    8,
    16,
    32
]

METERS = [
    (3, 4),
    (4, 4),
    (5, 4),
    (6, 4),
    (7, 4)
]

DURATION_TO_QUARTERS = {

    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,

    "half_dotted": 3.0,
    "quarter_dotted": 1.5,
    "eighth_dotted": 0.75,
    "sixteenth_dotted": 0.375,

    "half_triplet": 4 / 3,
    "quarter_triplet": 2 / 3,
    "eighth_triplet": 1 / 3,
    "sixteenth_triplet": 1 / 6
}

duration_names = sorted(
    DURATION_TO_QUARTERS.keys(),
    key=len,
    reverse=True
)

KEY_TO_SEMITONES = {

    "C": 0,
    "C#": 1,
    "Db": 1,

    "D": 2,

    "D#": 3,
    "Eb": 3,

    "E": 4,

    "F": 5,

    "F#": 6,
    "Gb": 6,

    "G": 7,

    "G#": 8,
    "Ab": 8,

    "A": 9,

    "A#": 10,
    "Bb": 10,

    "B": 11
}

SEQ_LEN = 64

model = load_model("event_model.keras")

with open("event_to_int.pkl", "rb") as f:
    event_to_int = pickle.load(f)

with open("int_to_event.pkl", "rb") as f:
    int_to_event = pickle.load(f)

X_events = np.load("X_events.npy")

rest_indices = []

for idx, event in int_to_event.items():
    if event.startswith("REST_"):
        rest_indices.append(idx)


def generate_song(target_key, target_bpm, target_bars, time_num, time_denom, file_name):

    generated_events = []

    idx = np.random.randint(len(X_events))
    seed = X_events[idx].copy()
    TARGET_BARS = target_bars
    TARGET_BPM = target_bpm
    TARGET_KEY = target_key
    TIME_SIGNATURE_NUM = time_num
    TIME_SIGNATURE_DENOM = time_denom
    current_quarters = 0
    target_quarters = TARGET_BARS * 4 * TIME_SIGNATURE_NUM / TIME_SIGNATURE_DENOM

    while True:

        prediction = model.predict(seed.reshape(1, SEQ_LEN), verbose=0)[0]
        temperature = 0.9
        prediction = np.log(prediction + 1e-8)
        prediction /= temperature
        prediction = np.exp(prediction)
        prediction /= np.sum(prediction)

        for idx in rest_indices:
            prediction[idx] *= 0.5

        if len(generated_events) > 0 and generated_events[-1].startswith("REST_"):
            for idx in rest_indices:
                prediction[idx] *= 0.2

        prediction /= np.sum(prediction)
        song_end_idx = event_to_int["<SONG_END>"]
        prediction[song_end_idx] = 0
        prediction /= np.sum(prediction)

        k = 10
        top_idx = np.argsort(prediction)[-k:]
        top_probs = prediction[top_idx]
        top_probs /= np.sum(top_probs)
        event_idx = np.random.choice(top_idx, p=top_probs)
        event_name = int_to_event[event_idx]

        # if event_name == "<SONG_END>":
        #     break

        duration_name = None

        for dur in duration_names:

            if event_name.endswith("_" + dur):
                duration_name = dur
                break

        if duration_name is not None:
            current_quarters += (DURATION_TO_QUARTERS[duration_name])

        generated_events.append(event_name)

        if current_quarters >= target_quarters:
            break

        seed = np.append(seed[1:], event_idx)

    melody = stream.Stream()
    melody.insert(0, tempo.MetronomeMark(number=TARGET_BPM))
    melody.insert(0, meter.TimeSignature(f"{TIME_SIGNATURE_NUM}/{TIME_SIGNATURE_DENOM}"))

    for event in generated_events:

        pitch_name = None
        duration_name = None

        for dur in duration_names:

            suffix = "_" + dur

            if event.endswith(suffix):

                pitch_name = event[:-len(suffix)]
                duration_name = dur
                break

        if pitch_name is None:
            continue

        try:
            if pitch_name == "REST":
                n = note.Rest()
            else:
                n = note.Note(pitch_name)

            n.quarterLength = (DURATION_TO_QUARTERS[duration_name])
            melody.append(n)

        except Exception as e:
            print("Skipping:", pitch_name, duration_name, e)

    detected_key = melody.analyze("key")

    current_semitone = detected_key.tonic.pitchClass
    target_semitone = KEY_TO_SEMITONES[TARGET_KEY]
    transpose_amount = target_semitone - current_semitone

    if transpose_amount > 6:
        transpose_amount -= 12

    if transpose_amount < -6:
        transpose_amount += 12

    melody.transpose(transpose_amount, inPlace=True)
    melody.write("midi", fp=file_name)
    print(f"Saved {file_name}")


counter = 0

for target_key in KEYS:
    for bpm_min, bpm_max in BPM_RANGES:
        for target_bars in BARS_OPTIONS:
            for time_num, time_denom in METERS:
                target_bpm = random.randint(bpm_min, bpm_max)
                filename = f"{target_key}_{target_bpm}bpm_{target_bars}bars_{time_num}-{time_denom}"
                generate_song(target_key, target_bpm, target_bars, time_num, time_denom,
                              os.path.join(OUTPUT_DIR, filename + ".mid"))

                counter += 1
                print(counter,filename)

