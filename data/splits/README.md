# Splits

`ffpp_train.json`, `ffpp_val.json`, `ffpp_test.json` are the official FaceForensics++ splits,
copied unchanged from https://github.com/ondyari/FaceForensics (dataset/splits).

Each entry is a pair of original video ids, for example `["071", "054"]`.
For a pair `[a, b]` the split contains:
- original videos `a` and `b`
- manipulated videos `a_b` and `b_a` for every method (Deepfakes, Face2Face, FaceSwap, NeuralTextures)

Pairs: 360 train, 70 val, 70 test, which gives 720 / 140 / 140 original videos.
Both identities of a pair stay in the same split, so no identity appears in two splits.

Celeb-DF v2 is only used for testing, with the test list that ships with the dataset.
