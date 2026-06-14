import pickle
import numpy as np
from tensorflow.keras.models import load_model
from music21 import stream, meter
from music21 import note
from music21 import tempo
from music21 import key


SEQ_LEN = 64

model = load_model("event_model.keras")

with open("event_to_int.pkl", "rb") as f:
    event_to_int = pickle.load(f)

with open("int_to_event.pkl", "rb") as f:
    int_to_event = pickle.load(f)

generated_events = []

X_events = np.load("X_events.npy")

idx = np.random.randint(len(X_events))

seed = X_events[idx].copy()

rest_indices = []

for idx, event in int_to_event.items():

    if event.startswith("REST_"):
        rest_indices.append(idx)


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

duration_names = sorted(DURATION_TO_QUARTERS.keys(), key=len, reverse=True)

TARGET_BARS = int(input("Wprowadź liczbę taktów: "))
TARGET_BPM = int(input("Wprowadź tempo (BPM): "))
TARGET_KEY = input("Wprowadź tonację (C, C#, D, Eb...): ")
TIME_SIGNATURE_NUM = int(input("Wprowadź metrum (licznik): "))
TIME_SIGNATURE_DENOM = int(input("Wprowadź metrum (mianownik): "))
OUTPUT_FILE = input("Nazwa pliku: ")

current_quarters = 0
target_quarters = TARGET_BARS * 4 * TIME_SIGNATURE_NUM / TIME_SIGNATURE_DENOM

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


while True:

    prediction = model.predict(seed.reshape(1, SEQ_LEN),verbose=0)[0]
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
melody.insert(0,meter.TimeSignature(f"{TIME_SIGNATURE_NUM}/{TIME_SIGNATURE_DENOM}"))

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


print("\nDetected key:", detected_key.tonic.name, detected_key.mode)
current_semitone = detected_key.tonic.pitchClass
target_semitone = KEY_TO_SEMITONES[TARGET_KEY]
transpose_amount = (target_semitone - current_semitone) % 12

melody.transpose(transpose_amount, inPlace=True)

print(f"\nGenerated events: {len(generated_events)}\n")
print(generated_events[:50])
print(f"\nGenerated quarters: {round(current_quarters, 2)}\n")

print("Original key:", detected_key.tonic.name, detected_key.mode)
print("Target tonic:", TARGET_KEY)
final_key = melody.analyze("key")
print("Final key:", final_key.tonic.name, final_key.mode)

tempo_marks = melody.getElementsByClass(tempo.MetronomeMark)

if tempo_marks:
    print("Final BPM:", tempo_marks[0].number)

melody.write("midi", fp=OUTPUT_FILE + ".mid")

print(f"Saved {OUTPUT_FILE}.mid")
