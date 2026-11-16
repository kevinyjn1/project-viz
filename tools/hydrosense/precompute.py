#!/usr/bin/env python3
"""Precompute every number shown on the HydroSense "under the hood" page.

Usage:
    git clone --depth 1 https://github.com/saideepk444/HydroSense /path/to/HydroSense
    python3 precompute.py /path/to/HydroSense ../../hydrosense/data

Everything comes from the repo itself: the WAV files in clean_data/ and data/,
and the saved models/ (svm_model.pkl, scaler.pkl, label_encoder.pkl).
Feature extraction is copied from tap_classifier.py / processing.ipynb
(librosa.load at sr=44100, 13 MFCCs, n_fft=2048, hop=512, n_mels=128,
then mean/std/max/min per coefficient = 52 numbers).

Needs: numpy, scipy, scikit-learn, librosa, joblib.
"""
import json, os, sys, warnings, wave
from collections import Counter
import numpy as np
import joblib, librosa, scipy.fft
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
REPO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '../../hydrosense/data')
os.makedirs(OUT, exist_ok=True)

SR, N_MFCC, N_FFT, HOP, N_MELS = 44100, 13, 2048, 512, 128
ORDER = ['empty', '100ml', '300ml', '500ml', '700ml', '900ml']   # display order (by water level)


def r(a, d):
    """Round floats (recursively) for compact JSON."""
    a = np.asarray(a, dtype=float)
    if d == 0:
        return np.rint(a).astype(int).tolist()
    return np.round(a, d).tolist()


def extract_features(path):
    """Same as tap_classifier.extract_features, but also returns the pieces."""
    y, sr = librosa.load(path, sr=SR)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    feats = np.concatenate([mf.mean(1), mf.std(1), mf.max(1), mf.min(1)])
    return feats, y, mf


# ---------------------------------------------------------------- models
model = joblib.load(f'{REPO}/models/svm_model.pkl')
scaler = joblib.load(f'{REPO}/models/scaler.pkl')
le = joblib.load(f'{REPO}/models/label_encoder.pkl')
LE = list(le.classes_)
assert model.kernel == 'linear' and len(LE) == 6
nC = len(LE)
pairs = [(i, j) for i in range(nC) for j in range(i + 1, nC)]   # libsvm one-vs-one order

# ---------------------------------------------------------------- features of every clean recording
cd = f'{REPO}/clean_data'
files = sorted(f for f in os.listdir(cd) if f.endswith('.wav') and f.split('_')[0] in LE)
X = np.array([extract_features(f'{cd}/{f}')[0] for f in files])
y = np.array([f.split('_')[0] for f in files])
Xs = scaler.transform(X)

# ---------------------------------------------------------------- recover the training split
# Every support vector must be a training recording; the scaler's mean/variance pin down the rest.
D = ((model.support_vectors_[:, None, :] - Xs[None]) ** 2).sum(-1)
sv_files = set(D.argmin(1))
assert np.sqrt(D.min(1)).max() < 1e-3, 'support vectors do not match clean_data features'
n, nt = len(files), int(scaler.n_samples_seen_)
A = np.vstack([X.T / scaler.scale_[:, None], (X ** 2).T / scaler.scale_[:, None] ** 2, np.ones((1, n))])
b = np.concatenate([nt * scaler.mean_ / scaler.scale_, nt * (scaler.var_ + scaler.mean_ ** 2) / scaler.scale_ ** 2, [nt]])
tol = np.concatenate([np.full(2 * X.shape[1], 0.05), [0]])
lb = np.zeros(n); lb[list(sv_files)] = 1
res = milp(np.zeros(n), constraints=LinearConstraint(A, b - tol, b + tol), integrality=np.ones(n), bounds=Bounds(lb, np.ones(n)))
train = np.rint(res.x).astype(int)
alt = milp(np.zeros(n), constraints=[LinearConstraint(A, b - tol, b + tol), LinearConstraint((1 - train)[None].astype(float), 1, np.inf)],
           integrality=np.ones(n), bounds=Bounds(lb, np.ones(n)))
split_unique = alt.status == 2
print('training recordings recovered:', train.sum(), 'unique:', split_unique,
      'max mean error:', np.abs(X[train == 1].mean(0) - scaler.mean_).max())

pred = le.inverse_transform(model.predict(Xs))
test = train == 0
cm = confusion_matrix(y[test], pred[test], labels=ORDER)
print('held-out accuracy', (pred[test] == y[test]).sum(), '/', test.sum())

# recordings from the day before (data/), never used for this model
dd = f'{REPO}/data'
dfiles = sorted(f for f in os.listdir(dd) if f.endswith('.wav') and f.split('_')[0] in LE)
Xd = np.array([extract_features(f'{dd}/{f}')[0] for f in dfiles])
yd = np.array([f.split('_')[0] for f in dfiles])
pd_ = le.inverse_transform(model.predict(scaler.transform(Xd)))
cm_d = confusion_matrix(yd, pd_, labels=ORDER)
print('data/ accuracy', (pd_ == yd).sum(), '/', len(yd))

# ---------------------------------------------------------------- PCA map
pca = PCA(n_components=2).fit(Xs)
P = pca.transform(Xs)

# ---------------------------------------------------------------- mel filterbank (same one librosa.feature.mfcc uses)
mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
fft_hz = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
mel_pts = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SR / 2)  # edges: lo, centre, hi
filters = [[float(mel_pts[k]), float(mel_pts[k + 1]), float(mel_pts[k + 2]), float(mel_fb[k].max())] for k in range(N_MELS)]


def decision_parts(xs):
    ovo = xs @ model.coef_.T + model.intercept_          # 15 pairwise values w.x+b
    votes = np.zeros(nC, int)
    for k, (i, j) in enumerate(pairs):
        votes[i if ovo[k] > 0 else j] += 1
    return ovo, votes


# check the hand-made one-vs-one maths against scikit-learn
model.decision_function_shape = 'ovo'
assert np.allclose(model.decision_function(Xs[:5]), Xs[:5] @ model.coef_.T + model.intercept_, atol=1e-6)
model.decision_function_shape = 'ovr'

# ---------------------------------------------------------------- pick recordings to show
picks = []
for c in ORDER:   # one correctly classified held-out recording per class, with a typical margin
    idx = [i for i in range(n) if test[i] and y[i] == c and pred[i] == c]
    margins = [np.sort(model.decision_function(Xs[i:i + 1])[0])[-1] - np.sort(model.decision_function(Xs[i:i + 1])[0])[-2] for i in idx]
    picks.append(idx[int(np.argsort(margins)[len(margins) // 2])])
miss = [i for i in range(n) if test[i] and pred[i] != y[i]]
picks.append(miss[0])

recs = []
for i in picks:
    f = files[i]
    feats, yy, mf = extract_features(f'{cd}/{f}')
    with wave.open(f'{cd}/{f}') as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(int)
        file_sr = w.getframerate()
    # waveform envelope for display: 600 min/max bins over the raw int16 samples
    nb = 600
    edges = np.linspace(0, len(raw), nb + 1).astype(int)
    env_min = [int(raw[a:b_].min()) for a, b_ in zip(edges[:-1], edges[1:])]
    env_max = [int(raw[a:b_].max()) for a, b_ in zip(edges[:-1], edges[1:])]
    # spectrogram pieces exactly as librosa.feature.mfcc builds them
    S = np.abs(librosa.stft(yy, n_fft=N_FFT, hop_length=HOP)) ** 2          # power, 1025 x frames
    M = mel_fb @ S
    Mdb = librosa.power_to_db(M)                                            # ref=1, top_db=80
    mf2 = scipy.fft.dct(Mdb, axis=0, type=2, norm='ortho')[:N_MFCC]
    assert np.allclose(mf2, mf, atol=1e-3)
    t = int(np.argmax(S.sum(0)))                                             # loudest frame
    ypad = np.pad(yy, N_FFT // 2)
    frame = ypad[t * HOP: t * HOP + N_FFT]
    spec_db = 10 * np.log10(np.maximum(S[:, t], 1e-10))
    xs = scaler.transform(feats[None])[0]
    ovo, votes = decision_parts(xs)
    ovr = model.decision_function(xs[None])[0]
    proba = model.predict_proba(xs[None])[0]
    pcls = LE[int(model.predict(xs[None])[0])]
    # time-pooled mel spectrogram for display (mean of each 2 frames, in dB)
    T = Mdb.shape[1]
    Mp = np.array([Mdb[:, k:k + 2].mean(1) for k in range(0, T, 2)]).T
    rid = f.replace('.wav', '')
    recs.append({'id': rid, 'file': f, 'cls': y[i], 'pred': pcls})
    out = {
        'id': rid, 'file': 'clean_data/' + f, 'cls': y[i], 'heldOut': bool(test[i]),
        'fileSr': file_sr, 'nSamples': int(len(raw)), 'nResampled': int(len(yy)), 'dc': round(float(raw.mean()), 1),
        'env': {'min': env_min, 'max': env_max},
        'first': raw[:24].tolist(),                       # first int16 samples (for the UART bytes)
        'nFrames': int(T), 'frameIdx': t,
        'frame': r(frame[::2], 4),                        # every 2nd of the 2048 samples, float [-1,1]
        'spec': r(spec_db, 1),                            # 1025 bins, dB of power
        'melCol': r(Mdb[:, t], 1), 'melColRaw': r(10 * np.log10(np.maximum(M[:, t], 1e-10)), 1),
        'mfccCol': r(mf[:, t], 2),
        'melspec': r(Mp, 0), 'melPool': 2, 'melMax': float(Mdb.max()),
        'mfcc': r(mf, 1),
        'feat': r(feats, 3), 'scaled': r(xs, 3),
        'ovo': r(ovo, 4), 'votes': votes.tolist(), 'ovr': r(ovr, 4),
        'proba': r(proba, 4), 'pred': pcls,
    }
    with open(f'{OUT}/rec_{rid}.json', 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))

model_json = {
    'source': 'saideepk444/HydroSense models/*.pkl (scikit-learn ' + str(getattr(model, '_sklearn_version', '1.5.1')) + ')',
    'order': ORDER, 'leClasses': LE, 'pairs': pairs,
    'kernel': model.kernel, 'C': model.C, 'dfShape': 'ovr', 'nSupport': model.n_support_.tolist(),
    'coef': r(model.coef_, 4), 'intercept': r(model.intercept_, 4),
    'mean': r(scaler.mean_, 4), 'scale': r(scaler.scale_, 4), 'nTrain': nt,
    'filters': [[round(a, 1), round(b_, 1), round(c, 1), round(h, 6)] for a, b_, c, h in filters],
    'recs': recs, 'hero': recs[4]['id'],
    'pca': {'var': r(pca.explained_variance_ratio_, 4),
            'pts': [[round(float(P[k, 0]), 2), round(float(P[k, 1]), 2), ORDER.index(y[k]), ORDER.index(pred[k]), int(test[k]), files[k].replace('.wav', '')] for k in range(n)]},
    'heldOut': {'n': int(test.sum()), 'correct': int((pred[test] == y[test]).sum()), 'cm': cm.tolist(),
                'splitUnique': bool(split_unique), 'counts': {c: int((y == c).sum()) for c in ORDER}},
    'otherDay': {'n': len(yd), 'correct': int((pd_ == yd).sum()), 'cm': cm_d.tolist()},
}
with open(f'{OUT}/model.json', 'w') as fh:
    json.dump(model_json, fh, separators=(',', ':'))
print('picked', [(x['id'], x['cls'], x['pred']) for x in recs])
