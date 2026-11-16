#!/usr/bin/env python3
"""Re-run AmbientMNIST's on-device forward pass in Python, bit for bit.

What this follows (read these files to check):
  - apps/my_mnist/src/main.c          : layer order, buffers, which bias is used
  - apps/my_mnist/src/headers/*.h     : the weights and the test image, as F_LIT(x)
  - SONIC libfixed (fixed.h)          : 16-bit fixed point, 5 fractional bits
  - SONIC libdnn  (task_dm_conv.c, task_dm_mul.c, task_ds_add.c,
                   task_dm_add.c, nonlinear.c) : exact multiply/add order

Usage:
  python3 precompute.py /path/to/AmbientMNIST/ambientMNIST  ../../ambient-mnist/data  [mnist.npz]

  mnist.npz is the file Keras downloads for mnist.load_data()
  (https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz,
   sha256 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1).
  It is only needed for the README demo image (test image #3, a "0").

Only numpy is needed. The headers are parsed as text; no pickle is loaded.
"""
import json, re, sys, os
import numpy as np

F_N = 5            # LIBFIXED_FRAC_BITWIDTH default in SONIC's libfixed
F_ONE = 1 << F_N   # 32
F_K = 1 << (F_N - 1)  # 16, used for round-to-nearest in f_mul

# The device printout shown in the README (ambientMNIST/output1.png)
DEVICE_LOGITS = [247, -175, -81, -86, -58, -77, 89, -75, 26, -12]

overflow = {"mul": 0, "add": 0}

def wrap16(x):
    """MSP430 'int' is 16 bits: keep the low 16 bits, signed."""
    x = int(x)
    y = ((x + 32768) & 0xFFFF) - 32768
    return y

def f_lit(s):
    # #define F_LIT(f) (fixed)(f * F_ONE)   -> C truncates toward zero
    v = float(s) * F_ONE
    return wrap16(int(v))

def f_mul(a, b):
    # signed int tmp = a * b; tmp += F_K; tmp >>= F_N;   (all 16-bit int)
    p = a * b
    if p != wrap16(p): overflow["mul"] += 1
    t = wrap16(p)
    t = wrap16(t + F_K)
    return t >> F_N          # arithmetic shift on a signed value

def f_add(a, b):
    s = a + b
    if s != wrap16(s): overflow["add"] += 1
    return wrap16(s)

def read_header(path):
    txt = open(path).read()
    out = {}
    for m in re.finditer(r'fixed\s+(\w+)((?:\[\d+\])+)\s*=\s*\{(.*?)\};', txt, re.S):
        name, dims, body = m.group(1), m.group(2), m.group(3)
        shape = [int(d) for d in re.findall(r'\[(\d+)\]', dims)]
        raw = re.findall(r'F_LIT\(([^)]*)\)', body)
        assert len(raw) == int(np.prod(shape)), (name, len(raw), shape)
        out[name] = {"shape": shape, "raw": [float(r) for r in raw],
                     "q": np.array([f_lit(r) for r in raw], dtype=np.int64).reshape(shape)}
    return out

def conv_layer(src, w, b):
    """task_d_conv -> task_dm_conv (+ task_ds_add for the bias).
    For each filter, for each single weight (k, l, n) in order, sweep every output
    pixel: acc = f_mul(w, x) (+ previous acc). Each product is rounded on its own."""
    F, C, KH, KW = w.shape
    _, H, W = src.shape
    OH, OW = H - KH + 1, W - KW + 1
    acc_out = np.zeros((F, OH, OW), dtype=np.int64)
    out = np.zeros((F, OH, OW), dtype=np.int64)
    for f in range(F):
        acc = [[0] * OW for _ in range(OH)]
        first = True
        for k in range(C):
            for l in range(KH):
                for n in range(KW):
                    wv = int(w[f, k, l, n])
                    for i in range(OH):
                        row = src[k, i + l]
                        for j in range(OW):
                            p = f_mul(wv, int(row[j + n]))
                            acc[i][j] = p if first else f_add(p, acc[i][j])
                    first = False
        for i in range(OH):
            for j in range(OW):
                acc_out[f, i, j] = acc[i][j]
                out[f, i, j] = f_add(acc[i][j], int(b[f]))   # task_ds_add
    return acc_out, out

def relu(x):
    return np.where(x < 0, 0, x)

def pool2(x):
    F, H, W = x.shape
    out = np.zeros((F, H // 2, W // 2), dtype=np.int64)
    arg = np.zeros((F, H // 2, W // 2), dtype=np.int64)  # which of the 4 won (0..3)
    for f in range(F):
        for j in range(0, H, 2):
            for k in range(0, W, 2):
                best = x[f, j, k]; bi = 0   # 'max = src[i,j,k]' then strict '<'
                for l in range(2):
                    for m in range(2):
                        v = x[f, j + l, k + m]
                        if best < v: best = v; bi = l * 2 + m
                out[f, j // 2, k // 2] = best; arg[f, j // 2, k // 2] = bi
    return out, arg

def dense(W, x, b=None):
    """task_d_fc -> task_dm_mul: for k in columns, acc[i] = f_mul(W[i,k], x[k]) + acc[i]."""
    R, Cc = W.shape
    acc = [0] * R
    for k in range(Cc):
        for i in range(R):
            p = f_mul(int(W[i, k]), int(x[k]))
            acc[i] = p if k == 0 else f_add(p, acc[i])
    acc = np.array(acc, dtype=np.int64)
    if b is None:
        return acc, acc
    return acc, np.array([f_add(int(a), int(bb)) for a, bb in zip(acc, b)], dtype=np.int64)

def forward(x, q):
    """x: (1,28,28) fixed-point ints. Returns every buffer the device fills, in order."""
    c1acc, c1 = conv_layer(x, q["conv1_w"], q["conv1_b"])   # main.c state 1: b2 -> b1 (16,24,24)
    r1 = relu(c1)                                            # state 2: b1 -> b2
    p1, p1arg = pool2(r1)                                    # state 3: b2 -> b1 (16,12,12)
    c2acc, c2 = conv_layer(p1, q["conv2_w"], q["conv2_b"])  # state 4: b1 -> b2 (16,8,8)
    r2 = relu(c2)                                            # state 5: b2 -> b1
    p2, p2arg = pool2(r2)                                    # state 6: b1 -> b2 (16,4,4)
    flat = p2.reshape(-1)                                    # state 7: reshape to 256 (channel-major)
    f1acc, f1 = dense(q["fc1_w"], flat, None)                # main.c: b_ptr = NULL -> no fc1 bias
    r3 = relu(f1)                                            # state 8
    f2acc, logits = dense(q["fc2_w"], r3, q["fc2_b"])       # state 9 (with fc2 bias)
    pred = 0; best = 0                                       # task_finish: max starts at 0, strict >
    for i, v in enumerate(logits):
        if v > best: best = v; pred = i
    L = lambda a: [int(v) for v in np.asarray(a).reshape(-1)]
    return {"input": L(x), "conv1": L(c1), "relu1": L(r1), "pool1": L(p1),
            "conv2": L(c2), "relu2": L(r2), "pool2": L(p2),
            "fc1": L(f1), "relu3": L(r3), "fc2_acc": L(f2acc), "logits": L(logits),
            "prediction": pred}

def float_forward(xf, h):
    g = lambda n: np.array(h[n]["raw"]).reshape(h[n]["shape"])
    def conv(x, w, b):
        F, C, KH, KW = w.shape; _, H, W = x.shape
        o = np.zeros((F, H - KH + 1, W - KW + 1))
        for f in range(F):
            for i in range(H - KH + 1):
                for j in range(W - KW + 1):
                    o[f, i, j] = np.sum(w[f] * x[:, i:i + KH, j:j + KW]) + b[f]
        return o
    pool = lambda x: x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2] // 2, 2).max(axis=(2, 4))
    a = pool(np.maximum(conv(xf, g("conv1_w"), g("conv1_b")), 0))
    a = pool(np.maximum(conv(a, g("conv2_w"), g("conv2_b")), 0))
    v = np.maximum(g("fc1_w") @ a.reshape(-1) + g("fc1_b"), 0)
    return [round(float(t), 3) for t in (g("fc2_w") @ v + g("fc2_b"))]

def main(repo, out_dir, mnist_npz):
    hd = os.path.join(repo, "apps/my_mnist/src/headers")
    h = {}
    for n in ["conv1", "conv2", "fc1", "fc2", "input"]:
        h.update(read_header(os.path.join(hd, n + ".h")))
    q = {k: v["q"] for k, v in h.items()}

    # Optional cross-check against the compiled firmware in the repo.
    elf = os.path.join(repo, "apps/my_mnist/bld/gcc/my_mnist.out")
    elf_ok = None
    firmware = None
    if os.path.exists(elf):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from elf_arrays import read_elf_arrays
        e = read_elf_arrays(elf)
        elf_ok = all(list(q[k].reshape(-1)) == e[k] for k in e)
        from elf_arrays import symbol_sizes
        firmware = symbol_sizes(elf, [".text", "layer_buffers", "mat_buffers", "data_src", "data_dest", "data_size"])
        print("firmware sizes:", firmware)
        print("firmware arrays == header arrays:", elf_ok)

    runs = {}
    # Run 1: the image baked into the committed headers / firmware (an MNIST test "1").
    runs["one"] = dict(forward(q["input"], q), label=1,
        source="apps/my_mnist/src/headers/input.h (the image compiled into the committed firmware; it equals MNIST test image #5)",
        float_logits=float_forward(np.array(h["input"]["raw"]).reshape(1, 28, 28), h), device_logits=None)
    # Run 2: the README demo image. MNIST test image #3 (a "0"), the index the notebook saves.
    if mnist_npz:
        d = np.load(mnist_npz, allow_pickle=False)
        xf = (d["x_test"][3].astype("float32") / 255.0)
        xq = np.array([f_lit(repr(float(a))) for a in xf.reshape(-1).tolist()]).reshape(1, 28, 28)
        for r in range(28): print(" ".join(str(int(v)) for v in xq[0, r]))
        r0 = forward(xq, q)
        runs["zero"] = dict(r0, label=int(d["y_test"][3]),
            source="MNIST test image #3 (x_test[3] in the notebook), quantized with F_LIT; matches the 28x28 dump the device printed (ambientMNIST/input1.png)",
            float_logits=float_forward(xf.reshape(1, 28, 28).astype(np.float64), h), device_logits=DEVICE_LOGITS)
        r0["match"] = r0["logits"] == DEVICE_LOGITS
        runs["zero"]["match"] = r0["match"]
        # The notebook's own Python emulator printed logits for this image (with the weights of
        # the training run in that notebook session, which is not the one committed in headers/).
        try:
            nb = json.load(open(os.path.join(repo, "scripts/AmbientMNIST_Script.ipynb")))
            for cell in nb["cells"]:
                for o in cell.get("outputs", []):
                    t = "".join(o.get("text") or o.get("data", {}).get("text/plain") or "")
                    m = re.search(r"Logits: \[([^\]]*)\]", t)
                    if m:
                        runs["zero"]["notebook_logits"] = [float(v) for v in m.group(1).split(",")]
        except Exception as e:
            print("could not read notebook logits:", e)
        print("notebook emulator logits x32:", [v * 32 for v in runs["zero"].get("notebook_logits", [])])
        print("demo zero ours  :", r0["logits"], "pred", r0["prediction"])
        print("device printout :", DEVICE_LOGITS, "match:", r0["match"])

    L = lambda a: [int(v) for v in np.asarray(a).reshape(-1)]
    nbytes = {k: 2 * int(q[k].size) for k in ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b", "input"]}
    weights = {
        "format": {"bits": 16, "frac_bits": F_N, "one": F_ONE},
        "conv1_w": L(q["conv1_w"]), "conv1_b": L(q["conv1_b"]),
        "conv1_w_trained": [round(v, 6) for v in h["conv1_w"]["raw"]],
        "conv2_w": L(q["conv2_w"]), "conv2_b": L(q["conv2_b"]),
        "fc1_w": L(q["fc1_w"]), "fc1_b_unused": L(q["fc1_b"]),
        "fc2_w": L(q["fc2_w"]), "fc2_b": L(q["fc2_b"]),
        "bytes": nbytes,
        "zero_weights": {k: int(np.sum(q[k] == 0)) for k in ["conv1_w", "conv2_w", "fc1_w", "fc2_w"]},
        "firmware_matches_headers": elf_ok,
        "firmware_bytes": firmware,
        "overflow_events": overflow,
        "notebook_test_accuracy": 0.9883,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights, f, separators=(",", ":"))
    for k, r in runs.items():
        with open(os.path.join(out_dir, "run-%s.json" % k), "w") as f:
            json.dump(r, f, separators=(",", ":"))
    print("one :", runs["one"]["logits"], "pred", runs["one"]["prediction"], "float", runs["one"]["float_logits"])
    if "zero" in runs: print("zero float:", runs["zero"]["float_logits"])
    print("overflow:", overflow, "zeros:", weights["zero_weights"], "bytes:", nbytes)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
