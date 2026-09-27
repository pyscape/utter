#!/usr/bin/env python3
"""A NeMo TitaNet speaker model's ONNX export, converted once into the directory the crate reads.

    python scripts/titanet_convert.py nemo_en_titanet_small.onnx OUT_DIR

Needs onnx and numpy; the crate reads the result with neither. Writes OUT_DIR/titanet.conf and
OUT_DIR/weights.bin.

Every initializer is named by its NeMo module path, taken from the output names of the nodes
that read it (`/encoder/encoder/encoder.1/mconv.5/conv/Conv_output_0` is
`encoder.encoder.1.mconv.5.conv`), never by node order, and the graph is checked to be the
topology the crate implements: a first block of one separable convolution and
squeeze-excitation, blocks of separable convolutions with a residual, a last block like the
first, attentive statistics pooling and the embedding layer. The training classifier after
`embs` is dropped. Weights with 0 < |w| < 1e-15 are zeroed and counted. Depthwise kernels are
stored transposed, `[K, C]`.

weights.bin: the magic `UTTERTN\\0`, u32 version 1, u32 tensor count, then per tensor a u32
name length, the UTF-8 name, u32 rank, u32 dims, and a u64 offset from the start of the file;
the data follows, each tensor little-endian f32 at an offset that is a multiple of 64.
"""

import argparse
import hashlib
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

MAGIC = b"UTTERTN\0"
VERSION = 1
ALIGN = 64
TINY = 1e-15
# The front end the crate implements; a model whose metadata asks for another is refused.
FRONT_END = {
    "sample_rate": "16000",
    "feature_normalize_type": "per_feature",
    "window_size_ms": "25",
    "window_stride_ms": "10",
    "window_type": "hann",
    "feat_dim": "80",
}
# Sub-blocks per encoder block; the middle blocks carry a residual.
BLOCKS = (1, 3, 3, 3, 1)

type F32 = npt.NDArray[np.float32]


def fail(msg: str) -> NoReturn:
    sys.exit(f"refused: {msg}")


def module_path(output: str) -> str:
    """`/encoder/encoder/encoder.1/mconv.0/conv/Conv_output_0` -> `encoder.encoder.1.mconv.0.conv`:
    the exporter scopes a ModuleList child as `parent.index` beneath `parent`."""
    segs = output.strip("/").split("/")[:-1]
    out: list[str] = []
    for s in segs:
        if "." in s and out and s.rsplit(".", 1)[0] == out[-1]:
            out[-1] = s
        else:
            out.append(s)
    return ".".join(out)


class Graph:
    def __init__(self, model: Any) -> None:
        from onnx import numpy_helper

        self.init: dict[str, F32] = {
            t.name: numpy_helper.to_array(t).astype(np.float32) for t in model.graph.initializer
        }
        self.nodes = list(model.graph.node)
        self.producer = {o: n for n in self.nodes for o in n.output}
        self.by_path: dict[str, Any] = {}
        for n in self.nodes:
            if n.op_type in ("Conv", "MatMul", "BatchNormalization") and n.output:
                self.by_path[module_path(n.output[0])] = n

    def node(self, path: str, op: str) -> Any:
        n = self.by_path.get(path)
        if n is None or n.op_type != op:
            fail(f"no {op} at {path}")
        return n

    def attr(self, n: Any, name: str, default: Any = None) -> Any:
        from onnx import helper

        for a in n.attribute:
            if a.name == name:
                return helper.get_attribute_value(a)
        return default

    def weight(self, n: Any, i: int, path: str, suffix: str) -> F32:
        name = n.input[i]
        if name not in self.init:
            fail(f"{path}: input {i} is not an initializer")
        if not name.startswith("onnx::") and name != f"{path}.{suffix}":
            fail(f"{path}: initializer {name} is not {path}.{suffix}")
        return self.init[name]

    def through(self, tensor: str, *ops: str) -> str:
        """The tensor `ops` (innermost last) were applied to, to arrive at `tensor`."""
        for op in ops:
            n = self.producer.get(tensor)
            if n is None or n.op_type != op:
                fail(f"{tensor} is not the output of {op}")
            tensor = n.input[0]
        return tensor

    def through_relu(self, tensor: str) -> str:
        """The output of the one ReLU that reads `tensor`."""
        relu = [n for n in self.nodes if n.op_type == "Relu" and n.input[0] == tensor]
        if len(relu) != 1:
            fail(f"{tensor} is not followed by one ReLU")
        return str(relu[0].output[0])

    def bn_input(self, path: str) -> str:
        return str(self.node(path, "BatchNormalization").input[0])


def convert(g: Graph) -> Iterator[tuple[str, F32]]:
    """The tensors the crate reads, in file order, the topology checked on the way."""
    prev = "audio_signal"
    width = int(FRONT_END["feat_dim"])
    for b, subs in enumerate(BLOCKS):
        base = f"encoder.encoder.{b}"
        block_in = prev
        x = prev
        cin = width
        for s in range(subs):
            dw_path, pw_path = f"{base}.mconv.{5 * s}.conv", f"{base}.mconv.{5 * s + 1}.conv"
            dw, pw = g.node(dw_path, "Conv"), g.node(pw_path, "Conv")
            if dw.input[0] != x or (len(dw.input) > 2 and dw.input[2]):
                fail(f"{dw_path} does not read the previous layer, or has a bias")
            w = g.weight(dw, 1, dw_path, "weight")
            k = w.shape[2]
            if (
                w.shape[:2] != (cin, 1)
                or g.attr(dw, "group") != cin
                or list(g.attr(dw, "pads")) != [(k - 1) // 2] * 2
                or k % 2 == 0
                or list(g.attr(dw, "strides", [1])) != [1]
                or list(g.attr(dw, "dilations", [1])) != [1]
            ):
                fail(f"{dw_path} is not a same-length depthwise convolution")
            yield f"{dw_path}.weight", np.ascontiguousarray(w[:, 0, :].T)
            if pw.input[0] != dw.output[0]:
                fail(f"{pw_path} does not read {dw_path}")
            pw_w = g.weight(pw, 1, pw_path, "weight")
            cout = pw_w.shape[0]
            if pw_w.shape != (cout, cin, 1) or g.attr(pw, "group", 1) != 1:
                fail(f"{pw_path} is not a pointwise convolution")
            yield f"{pw_path}.weight", np.ascontiguousarray(pw_w[:, :, 0])
            yield f"{pw_path}.bias", g.weight(pw, 2, pw_path, "bias")
            x = pw.output[0]
            cin = cout
            if s + 1 < subs:
                x = g.through_relu(x)
        se = f"{base}.mconv.{5 * (subs - 1) + 3}.fc"
        fc0, fc2 = g.node(f"{se}.0", "MatMul"), g.node(f"{se}.2", "MatMul")
        w0, w2 = g.weight(fc0, 1, f"{se}.0", "weight"), g.weight(fc2, 1, f"{se}.2", "weight")
        r = w0.shape[1]
        if w0.shape != (cin, r) or w2.shape != (r, cin):
            fail(f"{se} is not a squeeze-excitation over {cin} channels")
        if g.through(fc2.input[0], "Relu") != fc0.output[0]:
            fail(f"{se}.2 does not read ReLU of {se}.0")
        yield f"{se}.0.weight", w0
        yield f"{se}.2.weight", w2
        # The block's output: ReLU of the gated last layer, plus the residual in the middle.
        outs = [n for n in g.nodes if n.op_type == "Relu" and module_path(n.output[0]) == f"{base}.mout.fc.1"]
        if len(outs) != 1:
            fail(f"{base} has no single output ReLU")
        gated = outs[0].input[0]
        if 0 < b < len(BLOCKS) - 1:
            add = g.producer.get(gated)
            res_path = f"{base}.res.0.0.conv"
            res = g.node(res_path, "Conv")
            if add is None or add.op_type != "Add" or add.input[1] != res.output[0]:
                fail(f"{base}'s output is not the gated layer plus {res_path}")
            if res.input[0] != block_in:
                fail(f"{res_path} does not read the block's input")
            rw = g.weight(res, 1, res_path, "weight")
            if rw.shape != (cin, width, 1):
                fail(f"{res_path} is not a pointwise convolution from {width} to {cin}")
            yield f"{res_path}.weight", np.ascontiguousarray(rw[:, :, 0])
            yield f"{res_path}.bias", g.weight(res, 2, res_path, "bias")
            gated = add.input[0]
        mul = g.producer.get(gated)
        if mul is None or mul.op_type != "Mul" or g.through(mul.input[1], "Sigmoid", "Transpose") != fc2.output[0]:
            fail(f"{base}'s gate is not the sigmoid of {se}.2")
        prev = outs[0].output[0]
        width = cin
    yield from pooling(g, prev, width)


def batchnorm(g: Graph, path: str, x: str, c: int) -> Iterator[tuple[str, F32]]:
    bn = g.node(path, "BatchNormalization")
    if bn.input[0] != x or g.attr(bn, "training_mode", 0) != 0:
        fail(f"{path} is not an inference batchnorm of {x}")
    for i, part in enumerate(("weight", "bias", "running_mean", "running_var"), start=1):
        t = g.weight(bn, i, path, part)
        if t.shape != (c,):
            fail(f"{path}.{part} is not {c} wide")
        yield f"{path}.{part}", t
    yield f"{path}.epsilon", np.asarray([g.attr(bn, "epsilon", 1e-5)], dtype=np.float32)


def pooling(g: Graph, h: str, c: int) -> Iterator[tuple[str, F32]]:
    att0 = "decoder._pooling.attention_layer.0.conv_layer"
    n = g.node(att0, "Conv")
    cat = g.producer.get(n.input[0])
    if cat is None or cat.op_type != "Concat" or cat.input[0] != h:
        fail(f"{att0} does not read the encoder's output beside its statistics")
    w = g.weight(n, 1, att0, "weight")
    a = w.shape[0]
    if w.shape != (a, 3 * c, 1):
        fail(f"{att0} is not a pointwise convolution from {3 * c}")
    yield f"{att0}.weight", np.ascontiguousarray(w[:, :, 0])
    yield f"{att0}.bias", g.weight(n, 2, att0, "bias")
    yield from batchnorm(g, "decoder._pooling.attention_layer.0.bn", g.through_relu(n.output[0]), a)
    att2 = "decoder._pooling.attention_layer.2"
    n2 = g.node(att2, "Conv")
    w2 = g.weight(n2, 1, att2, "weight")
    if w2.shape != (c, a, 1):
        fail(f"{att2} is not a pointwise convolution from {a} to {c}")
    bn_out = g.node("decoder._pooling.attention_layer.0.bn", "BatchNormalization").output[0]
    if g.through(n2.input[0], "Tanh") != bn_out:
        fail(f"{att2} does not read tanh of the attention's batchnorm")
    yield f"{att2}.weight", np.ascontiguousarray(w2[:, :, 0])
    yield f"{att2}.bias", g.weight(n2, 2, att2, "bias")
    sm = [x for x in g.nodes if x.op_type == "Softmax"]
    if len(sm) != 1 or g.attr(sm[0], "axis") != 2:
        fail("the attention is not one softmax over time")
    yield from batchnorm(g, "decoder.emb_layers.0.0", g.bn_input("decoder.emb_layers.0.0"), 2 * c)
    fc = "decoder.emb_layers.0.1"
    nf = g.node(fc, "Conv")
    wf = g.weight(nf, 1, fc, "weight")
    e = wf.shape[0]
    if wf.shape != (e, 2 * c, 1) or nf.input[0] != g.node("decoder.emb_layers.0.0", "BatchNormalization").output[0]:
        fail(f"{fc} is not the embedding layer over the pooled statistics")
    embs = g.producer.get("embs")
    if embs is None or embs.op_type != "Squeeze" or embs.input[0] != nf.output[0]:
        fail(f"embs is not {fc}'s output")
    yield f"{fc}.weight", np.ascontiguousarray(wf[:, :, 0])
    yield f"{fc}.bias", g.weight(nf, 2, fc, "bias")


def write(out: Path, tensors: list[tuple[str, F32]]) -> None:
    table = bytearray(MAGIC + struct.pack("<II", VERSION, len(tensors)))
    for name, t in tensors:
        b = name.encode()
        table += struct.pack("<I", len(b)) + b + struct.pack("<I", t.ndim) + struct.pack(f"<{t.ndim}I", *t.shape)
        table += b"\0" * 8
    offsets, at = [], len(table)
    for _, t in tensors:
        at = -(-at // ALIGN) * ALIGN
        offsets.append(at)
        at += 4 * t.size
    pos = len(MAGIC) + 8
    for (name, t), off in zip(tensors, offsets, strict=True):
        pos += 4 + len(name.encode()) + 4 + 4 * t.ndim
        table[pos : pos + 8] = struct.pack("<Q", off)
        pos += 8
    with open(out, "wb") as f:
        f.write(table)
        for (_, t), off in zip(tensors, offsets, strict=True):
            f.write(b"\0" * (off - f.tell()))
            f.write(np.ascontiguousarray(t, dtype="<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", help="a NeMo TitaNet export, e.g. nemo_en_titanet_small.onnx")
    ap.add_argument("out", help="the model directory to write")
    args = ap.parse_args()
    import onnx

    src = Path(args.onnx)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    model = onnx.load(str(src))
    meta = {p.key: p.value for p in model.metadata_props}
    for k, v in FRONT_END.items():
        if meta.get(k) != v:
            fail(f"metadata {k} is {meta.get(k)!r}, not {v!r}")
    g = Graph(model)
    tensors = list(convert(g))
    zeroed = 0
    for i, (name, t) in enumerate(tensors):
        tiny = (np.abs(t) < TINY) & (t != 0)
        zeroed += int(tiny.sum())
        tensors[i] = (name, np.where(tiny, np.float32(0), t).astype(np.float32))
    dim = tensors[-1][1].shape[0]
    if meta.get("output_dim") != str(dim):
        fail(f"metadata output_dim is {meta.get('output_dim')!r}, the embedding layer gives {dim}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "weights.bin", tensors)
    params = sum(t.size for _, t in tensors)
    conf = {
        "architecture": "titanet",
        **FRONT_END,
        "dim": str(dim),
        "source": src.name,
        "source_sha256": sha,
        "parameters": str(params),
        "zeroed": str(zeroed),
    }
    (out / "titanet.conf").write_text("".join(f"{k}={v}\n" for k, v in conf.items()))
    print(f"{len(tensors)} tensors, {params} parameters, {zeroed} zeroed below {TINY:g}; wrote {out}")


if __name__ == "__main__":
    main()
