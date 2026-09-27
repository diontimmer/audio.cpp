#!/usr/bin/env python3
"""Opt-in real-model LoRA check through the public C ABI. No model downloads.

Example:
  python tests/lora/c_api_smoke.py --library build/bin/libaudiocpp.dylib \
    --family stable_audio --model models/stable-audio-3-medium \
    --adapter /path/to/style.safetensors --backend metal --duration 4

Checks two sessions sharing one model, warm reuse, unadapted-session isolation,
zero-strength bypass, finite/non-silent audio and adapter-dependent output.
Identical audio is required only within one machine/backend/seed.
"""
import argparse
import array
import ctypes as C
import hashlib
import json
import math
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--family', choices=['stable_audio', 'ace_step'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--backend', default='cpu')
    p.add_argument('--duration', type=float, default=8)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--report')
    args = p.parse_args()
    lib = C.CDLL(str(Path(args.library).resolve()))
    ptr, string, integer = C.c_void_p, C.c_char_p, C.c_int
    class ModelConfig(C.Structure):
        _fields_ = [(name, string) for name in ['family_hint', 'config_id', 'weight_id', 'model_spec_override']]
    class Backend(C.Structure):
        _fields_ = [('backend', string), ('device', integer), ('threads', integer)]
    def bind(name, returns, *params):
        f = getattr(lib, 'audiocpp_' + name); f.restype = returns; f.argtypes = list(params)
        return f
    error = bind('last_error', string)
    registry_create = bind('registry_create', integer, string, C.POINTER(ptr))
    registry_free = bind('registry_free', None, ptr)
    model_load = bind('model_load', integer, ptr, string, C.POINTER(ModelConfig), ptr, C.POINTER(ptr))
    model_free = bind('model_free', None, ptr)
    options_create = bind('options_create', ptr)
    options_set = bind('options_set', integer, ptr, string, string)
    options_free = bind('options_free', None, ptr)
    session_create = bind('session_create', integer, ptr, string, string, C.POINTER(Backend), ptr, C.POINTER(ptr))
    session_free = bind('session_free', None, ptr)
    request_create = bind('request_create', ptr)
    request_free = bind('request_free', None, ptr)
    request_text = bind('request_set_text', integer, ptr, string, string)
    request_option = bind('request_set_option', integer, ptr, string, string)
    session_run = bind('session_run', integer, ptr, ptr, C.POINTER(ptr))
    result_free = bind('result_free', None, ptr)
    result_audio = bind('result_audio', integer, ptr, C.POINTER(C.POINTER(C.c_float)),
                        C.POINTER(C.c_size_t), C.POINTER(integer), C.POINTER(integer))
    def check(status):
        if status: raise RuntimeError(error().decode())
    registry, model = ptr(), ptr()
    sessions = []
    request = None
    records = []
    try:
        check(registry_create(None, C.byref(registry)))
        config = ModelConfig(args.family.encode(), None, None, None)
        check(model_load(registry, str(Path(args.model).resolve()).encode(), C.byref(config), None, C.byref(model)))
        backend = Backend(args.backend.encode(), 0, 4)
        request = request_create()
        check(request_text(request, b'Instrumental electronic music, melodic arpeggios, lively percussion, no vocals', None))
        for key, value in {'duration_seconds': args.duration, 'num_inference_steps': args.steps,
                           'seed': 1234, 'rewrite_caption': 'false'}.items():
            if key == 'rewrite_caption' and args.family != 'ace_step': continue
            check(request_option(request, key.encode(), str(value).encode()))
        def session(strength):
            opts = options_create()
            try:
                # Release GPU graphs between runs to bound the two-session test.
                check(options_set(opts, (args.family+'.mem_saver').encode(), b'true'))
                if strength is not None:
                    check(options_set(opts, (args.family+'.lora').encode(), str(Path(args.adapter).resolve()).encode()))
                    check(options_set(opts, (args.family+'.lora_strength').encode(), str(strength).encode()))
                value = ptr()
                check(session_create(model, b'gen', b'offline', C.byref(backend), opts, C.byref(value)))
                sessions.append(value)
                return value
            finally: options_free(opts)
        def run(s, label):
            result = ptr(); start = time.monotonic()
            try:
                check(session_run(s, request, C.byref(result)))
                samples = C.POINTER(C.c_float)(); frames=C.c_size_t(); rate=integer(); channels=integer()
                check(result_audio(result, C.byref(samples), C.byref(frames), C.byref(rate), C.byref(channels)))
                raw = C.string_at(samples, frames.value*channels.value*4)
                values=array.array('f'); values.frombytes(raw)
                if not values or not all(math.isfinite(v) for v in values): raise AssertionError('empty or non-finite audio')
                peak=max(abs(v) for v in values)
                if peak < 1e-7: raise AssertionError('silent audio')
                duration=frames.value/rate.value
                if abs(duration-args.duration) > .1: raise AssertionError(f'wrong duration: {duration}')
                record=dict(label=label, seconds=time.monotonic()-start, duration=duration,
                            frames=frames.value, sample_rate=rate.value, channels=channels.value,
                            peak=peak, sha256=hashlib.sha256(raw).hexdigest())
                records.append(record);print(json.dumps(record), flush=True)
                return record['sha256']
            finally:
                if result: result_free(result)
        baseline=session(None)
        off=run(baseline,'baseline')
        adapted=session(1)
        on=run(adapted,'adapted')
        if off==on: raise AssertionError('adapter did not affect audio')
        if run(adapted,'adapted_reuse') != on: raise AssertionError('warm session output changed')
        session_free(adapted);sessions.remove(adapted)
        if run(baseline,'baseline_after_adapter') != off: raise AssertionError('adapter mutated shared base')
        zero=session(0)
        if run(zero,'zero_strength') != off: raise AssertionError('disabled adapter altered audio')
        report=dict(family=args.family, backend=args.backend, passed=True, runs=records)
        if args.report: Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
        print('C API LoRA smoke passed',flush=True)
    finally:
        for s in reversed(sessions): session_free(s)
        if request: request_free(request)
        if model: model_free(model)
        if registry: registry_free(registry)

if __name__ == '__main__': main()
