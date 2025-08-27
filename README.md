# Face_Recognition-KNPC-
Review system for KNPC
A lightweight, offline desktop app (Python + OpenCV + face_recognition + PyQt) that auto-identifies a person from a local face dataset and instantly opens a review screen. The matched person’s profile image is shown, they select a rating, add a short comment, sign on a digital signature pad, and submit. All data is stored locally on the filesystem.

Key points

Real-time face detection and matching against a prepared dataset.

Auto-loads the person’s profile image and displays a clean review UI (rating + comment + signature).

Signature is captured as a PNG; an optional camera snapshot can be saved for audit.

Submissions are saved as JSON files (one per entry).

Fully offline and privacy-friendly; nothing leaves the machine.

Typical flow

Camera sees a face → system matches it to the nearest known identity.

Matched profile and name are shown with the review form and signature pad.

On submit, the app writes a JSON record (rating, comment, person ID, timestamps, file paths) and saves the signature PNG (and optional snapshot) to local folders.
