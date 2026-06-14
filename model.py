from mido import MidiFile
from enum import Enum
from music21 import converter
from music21 import note
from music21 import chord
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import glob
import pickle


class Duration(Enum):

    WHOLE = (0, 4.0)
    HALF = (1, 2.0)
    QUARTER = (2, 1.0)
    EIGHTH = (3, 0.5)
    SIXTEENTH = (4, 0.25)
    HALF_DOTTED = (5, 3.0)
    QUARTER_DOTTED = (6, 1.5)
    EIGHTH_DOTTED = (7, 0.75)
    SIXTEENTH_DOTTED = (8, 0.375)
    HALF_TRIPLET = (9, 4/3)
    QUARTER_TRIPLET = (10, 2/3)
    EIGHTH_TRIPLET = (11, 1/3)
    SIXTEENTH_TRIPLET = (12, 1/6)


def load_midi_file(midi_file_path):
    try:
        return MidiFile(midi_file_path)
    except Exception as e:
        raise ValueError(f"Failed to load MIDI file: {e}")


def adjust_duration(duration):
    fractions = [
        # Nuty
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,

        # Nuty z kropką
        0.375,
        0.75,
        1.5,
        3.0,

        # Triole
        4/3,
        2/3,
        1/3,
        1/6

    ]
    # Najbardziej odpowiednia nuta
    closest_fraction = min(fractions, key=lambda x: abs(x - duration))

    return closest_fraction


def get_duration_name(duration_quarters):
    for duration in Duration:
        _, duration_value = duration.value
        if duration_value == duration_quarters:
            return duration.name.lower()

    return None


event_tokens = []

midi_files = glob.glob(
    "midi_20_30_single_instrument/*.mid"
)

print("Files found:", len(midi_files))

for path in midi_files:

    try:
        midi = load_midi_file(path)
        score = converter.parse(path)

        for i, track in enumerate(midi.tracks):
            note_on_count = sum(1 for msg in track if msg.type == 'note_on' and msg.velocity > 1)
            if note_on_count == 0:
                continue

        part = score.parts[0]
        previous_offset = 0

        for element in part.flatten().notesAndRests:

            current_offset = element.offset
            gap = current_offset - previous_offset

            if isinstance(element, note.Rest):
                duration_quarters = adjust_duration(element.duration.quarterLength)
                duration_name = get_duration_name(duration_quarters)
                event_tokens.append("REST_" + duration_name)
                previous_offset = (element.offset + element.duration.quarterLength)
                continue

            if isinstance(element, note.Note):
                pitch = element.pitch

            elif isinstance(element, chord.Chord):
                pitch = max(element.pitches)

            else:
                continue

            duration_quarters = adjust_duration(element.duration.quarterLength)

            note_str = pitch.nameWithOctave
            note_str = note_str.replace('-', 'b')

            duration_name = get_duration_name(duration_quarters)

            event_tokens.append(note_str + "_" + duration_name)

            previous_offset = (element.offset + element.duration.quarterLength)

        event_tokens.append("<SONG_END>")

    except Exception as e:
        print("Skipping:", path)
        print(e)


all_events = sorted(set(event_tokens))
event_to_int = {event: idx for idx, event in enumerate(all_events)}
int_to_event = {idx: event for event, idx in event_to_int.items()}
encoded = [event_to_int[event] for event in event_tokens]

SEQ_LEN = 64

X = []
y = []

for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i:i + SEQ_LEN])
    y.append(encoded[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

VOCAB_SIZE = len(all_events)

y_cat = to_categorical(y,num_classes=VOCAB_SIZE)

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(VOCAB_SIZE, activation="softmax")
])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X,
    y_cat,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop]
)

with open("event_to_int.pkl", "wb") as f:
    pickle.dump(event_to_int, f)

with open("int_to_event.pkl", "wb") as f:
    pickle.dump(int_to_event, f)

np.save("X_events.npy",X)
model.save("event_model.keras")

print("Event model saved")
print("Vocabulary size:", VOCAB_SIZE)
print(all_events[:20])
