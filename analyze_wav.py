#!/usr/bin/env python3
"""Compare audio characteristics of three WAV files."""

import wave
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        sw = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sr, n_channels, sw

def rms(x):
    return np.sqrt(np.mean(x**2))

def spectral_flatness(x, sr):
    X = np.abs(fft(x))[:len(x)//2]
    power = X**2 + 1e-20
    geo = np.exp(np.mean(np.log(power)))
    arith = np.mean(power)
    return geo / arith

def spectral_centroid(x, sr):
    X = np.abs(fft(x))[:len(x)//2]
    freqs = fftfreq(len(x), 1/sr)[:len(x)//2]
    power = X**2
    total = np.sum(power)
    if total < 1e-20:
        return 0.0
    return np.sum(freqs * power) / total

def zero_crossing_rate(x):
    return np.sum(np.abs(np.diff(np.sign(x))) > 0) / len(x)

def per_second_analysis(samples, sr):
    n_seconds = int(len(samples) / sr)
    results = []
    for i in range(n_seconds):
        chunk = samples[i*sr:(i+1)*sr]
        results.append({
            'second': i,
            'rms': rms(chunk),
            'spectral_flatness': spectral_flatness(chunk, sr),
            'spectral_centroid': spectral_centroid(chunk, sr),
            'zcr': zero_crossing_rate(chunk),
        })
    remaining = samples[n_seconds*sr:]
    if len(remaining) > sr // 4:
        results.append({
            'second': n_seconds,
            'rms': rms(remaining),
            'spectral_flatness': spectral_flatness(remaining, sr),
            'spectral_centroid': spectral_centroid(remaining, sr),
            'zcr': zero_crossing_rate(remaining),
        })
    return results

def analyze_file(path, name):
    print(f"\n{'='*70}")
    print(f"  {name}: {path}")
    print(f"{'='*70}")
    samples, sr, n_ch, sw = read_wav(path)
    duration = len(samples) / sr
    print(f"  Sample rate:    {sr} Hz")
    print(f"  Channels:       {n_ch}")
    print(f"  Sample width:   {sw} bytes ({sw*8} bit)")
    print(f"  Num samples:    {len(samples)}")
    print(f"  Duration:       {duration:.4f} s")
    print(f"  Overall RMS:    {rms(samples):.6f} ({20*np.log10(rms(samples)+1e-20):.1f} dBFS)")
    print(f"  Peak:           {np.max(np.abs(samples)):.6f} ({20*np.log10(np.max(np.abs(samples))+1e-20):.1f} dBFS)")
    ps = per_second_analysis(samples, sr)
    print(f"\n  Per-second analysis ({len(ps)} segments):")
    print(f"  {'Sec':>3} | {'RMS':>8} | {'Spec Flat':>10} | {'Spec Cent (Hz)':>14} | {'ZCR':>8}")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*10}-+-{'-'*14}-+-{'-'*8}")
    for p in ps:
        print(f"  {p['second']:3d} | {p['rms']:8.6f} | {p['spectral_flatness']:10.6f} | {p['spectral_centroid']:14.1f} | {p['zcr']:8.5f}")
    return samples, sr, ps

def cross_correlation_analysis(s1, s2, sr, name1, name2):
    print(f"\n{'='*70}")
    print(f"  Cross-correlation: {name1} vs {name2}")
    print(f"{'='*70}")
    min_len = min(len(s1), len(s2))
    a = s1[:min_len]
    b = s2[:min_len]
    a_norm = a - np.mean(a)
    b_norm = b - np.mean(b)
    denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
    if denom < 1e-20:
        overall_corr = 0.0
    else:
        overall_corr = np.sum(a_norm * b_norm) / denom
    print(f"  Overall correlation (zero lag): {overall_corr:.6f}")
    corr_full = signal.correlate(a_norm, b_norm, mode='full', method='fft')
    corr_full /= denom
    lags = np.arange(-min_len+1, min_len)
    max_idx = np.argmax(corr_full)
    max_corr = corr_full[max_idx]
    max_lag = lags[max_idx]
    print(f"  Max correlation:               {max_corr:.6f} at lag {max_lag} samples ({max_lag/sr*1000:.2f} ms)")
    n_seconds = min(int(len(a)/sr), int(len(b)/sr))
    print(f"\n  Per-second correlation ({n_seconds} segments):")
    print(f"  {'Sec':>3} | {'Corr':>10} | {'RMS_diff':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}")
    for i in range(n_seconds):
        ca = a[i*sr:(i+1)*sr]
        cb = b[i*sr:(i+1)*sr]
        ca_n = ca - np.mean(ca)
        cb_n = cb - np.mean(cb)
        d = np.sqrt(np.sum(ca_n**2) * np.sum(cb_n**2))
        if d < 1e-20:
            corr = 0.0
        else:
            corr = np.sum(ca_n * cb_n) / d
        rms_diff = rms(ca) - rms(cb)
        print(f"  {i:3d} | {corr:10.6f} | {rms_diff:+10.6f}")
    return overall_corr

def spectral_comparison(files_data):
    print(f"\n{'='*70}")
    print(f"  Spectral Difference Analysis (Average Power Spectrum)")
    print(f"{'='*70}")
    spectra = {}
    for name, (samples, sr) in files_data.items():
        freqs, psd = signal.welch(samples, sr, nperseg=min(2048, len(samples)//2))
        spectra[name] = (freqs, psd)
    bands = [
        ('Sub-bass',    20,   60),
        ('Bass',        60,   250),
        ('Low-mid',     250,  500),
        ('Mid',         500,  2000),
        ('Upper-mid',   2000, 4000),
        ('Presence',    4000, 6000),
        ('Brilliance',  6000, 10000),
        ('Air',         10000, 20000),
    ]
    names = list(files_data.keys())
    print(f"\n  Average power by frequency band (dB):")
    header = f"  {'Band':<12} | {'Freq Range':>12}"
    for n in names:
        header += f" | {n:>14}"
    print(header)
    print(f"  {'-'*12}-+-{'-'*12}" + "".join(f"-+-{'-'*14}" for _ in names))
    band_powers = {n: {} for n in names}
    for band_name, lo, hi in bands:
        row = f"  {band_name:<12} | {lo:>5}-{hi:>5}Hz"
        for name in names:
            freqs, psd = spectra[name]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                avg_power = np.mean(psd[mask])
                db = 10 * np.log10(avg_power + 1e-20)
                band_powers[name][band_name] = db
                row += f" | {db:14.2f}"
            else:
                band_powers[name][band_name] = None
                row += f" | {'N/A':>14}"
        print(row)

    print(f"\n  Spectral difference (pocket - reference) by band (dB):")
    if 'pocket' in band_powers and 'reference' in band_powers:
        for band_name, _, _ in bands:
            p = band_powers['pocket'].get(band_name)
            r = band_powers['reference'].get(band_name)
            if p is not None and r is not None:
                diff = p - r
                indicator = ""
                if diff > 2:
                    indicator = " << MORE ENERGY (potential metallic)"
                elif diff < -2:
                    indicator = " << LESS ENERGY"
                print(f"    {band_name:<12}: {diff:+.2f} dB{indicator}")

    print(f"\n  Spectral difference (moonshine - reference) by band (dB):")
    if 'moonshine' in band_powers and 'reference' in band_powers:
        for band_name, _, _ in bands:
            m = band_powers['moonshine'].get(band_name)
            r = band_powers['reference'].get(band_name)
            if m is not None and r is not None:
                diff = m - r
                indicator = ""
                if diff > 2:
                    indicator = " << MORE ENERGY"
                elif diff < -2:
                    indicator = " << LESS ENERGY"
                print(f"    {band_name:<12}: {diff:+.2f} dB{indicator}")

    print(f"\n  Spectral difference (pocket - moonshine) by band (dB):")
    if 'pocket' in band_powers and 'moonshine' in band_powers:
        for band_name, _, _ in bands:
            p = band_powers['pocket'].get(band_name)
            m = band_powers['moonshine'].get(band_name)
            if p is not None and m is not None:
                diff = p - m
                indicator = ""
                if diff > 2:
                    indicator = " << pocket has MORE (metallic?)"
                elif diff < -2:
                    indicator = " << pocket has LESS"
                print(f"    {band_name:<12}: {diff:+.2f} dB{indicator}")

def main():
    print("WAV Audio Characteristics Comparison")
    print("=" * 70)
    pocket_samples, pocket_sr, pocket_ps = analyze_file(
        "/home/desmond/actor/pocket.wav", "pocket (stateful decoder)")
    ref_samples, ref_sr, ref_ps = analyze_file(
        "/home/desmond/actor/pocket_reference.wav", "reference (correct but monotonous)")
    moon_samples, moon_sr, moon_ps = analyze_file(
        "/home/desmond/actor/moonshine.wav", "moonshine (sounds better)")

    print(f"\n{'='*70}")
    print(f"  Summary Comparison")
    print(f"{'='*70}")
    metrics = ['rms', 'spectral_flatness', 'spectral_centroid', 'zcr']
    metric_labels = ['Avg RMS', 'Avg Spec Flatness', 'Avg Spec Centroid (Hz)', 'Avg ZCR']
    for label, metric in zip(metric_labels, metrics):
        p_vals = [p[metric] for p in pocket_ps]
        r_vals = [p[metric] for p in ref_ps]
        m_vals = [p[metric] for p in moon_ps]
        fmt = '.6f' if metric != 'spectral_centroid' else '.1f'
        print(f"  {label:<24}: pocket={np.mean(p_vals):{fmt}}  ref={np.mean(r_vals):{fmt}}  moon={np.mean(m_vals):{fmt}}")

    cross_correlation_analysis(pocket_samples, ref_samples, pocket_sr, "pocket", "reference")
    cross_correlation_analysis(moon_samples, ref_samples, moon_sr, "moonshine", "reference")
    cross_correlation_analysis(pocket_samples, moon_samples, pocket_sr, "pocket", "moonshine")

    spectral_comparison({
        'pocket': (pocket_samples, pocket_sr),
        'reference': (ref_samples, ref_sr),
        'moonshine': (moon_samples, moon_sr),
    })

    print(f"\n{'='*70}")
    print(f"  INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print(f"  - Higher spectral centroid = brighter/more metallic sound")
    print(f"  - Higher spectral flatness = more noise-like (less tonal)")
    print(f"  - Higher ZCR = more high-frequency content")
    print(f"  - More energy in 2-6kHz = metallic/harsh quality")
    print(f"  - Cross-correlation near 1.0 = very similar waveforms")
    print(f"  - Cross-correlation near 0.0 = very different waveforms")
    print()

if __name__ == '__main__':
    main()
