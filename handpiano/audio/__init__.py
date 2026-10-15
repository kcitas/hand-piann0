"""Audio: sample-based instruments, a polyphonic mixer and the output stream.

Independent of the camera, the hand tracker and the UI: anything that can call
``AudioEngine.note_on(midi, velocity)`` (mouse, keyboard, MIDI file, detected
presses, tests) can play it.
"""
