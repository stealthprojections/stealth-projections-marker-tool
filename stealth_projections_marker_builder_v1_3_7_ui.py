
import os
import sys
import csv
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import librosa


APP_TITLE = "Stealth Projections • Laser Marker Grid Builder"
VERSION = "1.3.4"


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes}:{sec:06.3f}"


def safe_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _to_float_scalar(value) -> float:
    if isinstance(value, np.ndarray):
        return float(value.squeeze())
    return float(value)


def clean_bpm(bpm: float) -> float:
    bpm_rounded = round(float(bpm), 1)
    frac = bpm_rounded - int(bpm_rounded)
    if frac < 0:
        frac = 0.0
    if frac >= 0.75 or frac <= 0.25:
        return float(round(bpm_rounded))
    return bpm_rounded


def suggest_bpm_value(bpm: float) -> int:
    return int(round(float(bpm)))


def detect_bpm_simple(y, sr) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    return clean_bpm(_to_float_scalar(tempo))


def detect_bpm_candidates_simple(y, sr) -> list[float]:
    base = detect_bpm_simple(y, sr)
    raw = [base, base * 2.0, base / 2.0]

    cleaned = []
    for c in raw:
        if 60 <= c <= 220:
            cleaned.append(suggest_bpm_value(c))

    cleaned = sorted(set(cleaned))
    cleaned = sorted(cleaned, key=lambda x: (0 if 100 <= x <= 180 else 1, abs(x - 128)))
    return cleaned


def detect_anchor(y, sr) -> float:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=8,
        post_avg=8,
        delta=float(np.std(onset_env)) * 0.5 if len(onset_env) else 0.0,
        wait=6,
    )
    if len(peaks) == 0:
        return 0.0

    peak_times = [float(times[p]) for p in peaks]
    early = [t for t in peak_times if 0.0 <= t <= 12.0]
    if early:
        return early[0]
    return peak_times[0]


def detect_phrase_bars(y, sr, anchor, seconds_per_bar, duration, prefer_16_bar=False, phrase_mode="balanced"):
    bars_total = int(np.floor((duration - anchor) / seconds_per_bar)) + 1
    if bars_total <= 1:
        return [0]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)

    phrase_candidates = []
    if len(onset_env):
        threshold = np.percentile(onset_env, 90 if phrase_mode == "aggressive" else 93)
        for i, val in enumerate(onset_env):
            if val >= threshold:
                t = float(times[i])
                bar = round((t - anchor) / seconds_per_bar)
                if bar >= 0 and bar % 8 == 0:
                    phrase_candidates.append(bar)

    phrase_bars = {0}
    spacing = 16 if prefer_16_bar else 8
    for b in sorted(set(phrase_candidates)):
        if b % spacing == 0 or phrase_mode == "aggressive":
            phrase_bars.add(b)

    return sorted(phrase_bars)


def build_marker_rows(
    wav_path: str,
    bpm_override: float | None = None,
    use_anchor_finding: bool = False,
    include_beats: bool = True,
    include_bar_markers: bool = True,
    include_4bar_markers: bool = True,
    include_8bar_markers: bool = True,
    include_16bar_markers: bool = True,
    include_32bar_markers: bool = False,
    include_phrase_guessing: bool = False,
    prefer_16_bar_phrases: bool = False,
    phrase_mode: str = "balanced",
):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    bpm = float(bpm_override) if bpm_override and bpm_override > 0 else detect_bpm_simple(y, sr)
    bpm = clean_bpm(bpm)

    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * 4.0

    detected_anchor = detect_anchor(y, sr)
    final_anchor = detected_anchor if use_anchor_finding else 0.0

    # BEYOND-import tuned colors
    color_phrase = "00FF00"  # neon green
    color_32 = "FF00FF"      # magenta
    color_16 = "00FFFF"      # cyan
    color_8 = "FFFF00"       # yellow
    color_4 = "FF3300"       # hot red/orange
    color_bar = "909090"     # soft gray
    color_beat = "202020"    # dark gray

    phrase_bars = detect_phrase_bars(
        y=y,
        sr=sr,
        anchor=final_anchor,
        seconds_per_bar=seconds_per_bar,
        duration=duration,
        prefer_16_bar=prefer_16_bar_phrases,
        phrase_mode=phrase_mode,
    ) if include_phrase_guessing else [0]

    rows = [["#", "Name", "Start", "Color"]]
    idx = 1
    beat = 0

    while True:
        t = final_anchor + beat * seconds_per_beat
        if t > duration + 1e-9:
            break

        if beat % 4 == 0:
            bar = beat // 4
            marker = None

            if include_phrase_guessing and bar in phrase_bars:
                marker = ("PHRASE", color_phrase)
            elif include_32bar_markers and bar % 32 == 0:
                marker = ("32 BAR", color_32)
            elif include_16bar_markers and bar % 16 == 0:
                marker = ("16 BAR", color_16)
            elif include_8bar_markers and bar % 8 == 0:
                marker = ("8 BAR", color_8)
            elif include_4bar_markers and bar % 4 == 0:
                marker = ("4 BAR", color_4)
            elif include_bar_markers:
                marker = ("BAR", color_bar)

            if marker is not None:
                name, color = marker
                rows.append([idx, name, fmt_time(t), color])
                idx += 1
        else:
            if include_beats:
                beat_in_bar = (beat % 4) + 1
                rows.append([idx, f"BEAT {beat_in_bar}", fmt_time(t), color_beat])
                idx += 1

        beat += 1

    meta = {
        "bpm_used": bpm,
        "candidate_bpms": detect_bpm_candidates_simple(y, sr),
        "detected_anchor_seconds": detected_anchor,
        "use_anchor_finding": use_anchor_finding,
        "final_anchor_seconds": final_anchor,
        "duration_seconds": duration,
        "phrase_bars": phrase_bars if include_phrase_guessing else [],
        "sample_rate": sr,
    }
    return rows, meta


def analyze_track(wav_path: str):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    candidates = detect_bpm_candidates_simple(y, sr)
    anchor = detect_anchor(y, sr)
    return {
        "candidate_bpms": candidates,
        "anchor_seconds": anchor,
        "duration_seconds": duration,
        "sample_rate": sr,
    }


def write_csv(rows, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


class HoverButton(tk.Button):
    def __init__(self, master, *, normal_bg, hover_bg, disabled_bg="#1a2330", **kwargs):
        super().__init__(master, **kwargs)
        self.normal_bg = normal_bg
        self.hover_bg = hover_bg
        self.disabled_bg = disabled_bg
        self.configure(
            bg=normal_bg,
            activebackground=hover_bg,
            relief="flat",
            bd=0,
            cursor="hand2",
            highlightthickness=0,
        )
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event):
        if str(self["state"]) != "disabled":
            self.configure(bg=self.hover_bg)

    def _on_leave(self, _event):
        if str(self["state"]) != "disabled":
            self.configure(bg=self.normal_bg)

    def set_enabled(self, enabled: bool):
        self.configure(state=("normal" if enabled else "disabled"))
        self.configure(bg=(self.normal_bg if enabled else self.disabled_bg))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)

        # Set taskbar/window icon
        try:
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
            ico_path = os.path.join(base_path, "stealth_projections_icon.ico")
            png_path = os.path.join(base_path, "square_dark_black_neon_sci_fi_abstract_scene_over.png")

            if os.path.exists(ico_path):
                try:
                    self.iconbitmap(ico_path)
                except Exception:
                    pass

            if os.path.exists(png_path):
                try:
                    self._window_icon = tk.PhotoImage(file=png_path)
                    self.iconphoto(True, self._window_icon)
                except Exception:
                    pass
        except Exception:
            pass
        self.geometry("1180x760")
        self.minsize(1100, 720)
        self.configure(bg="#05080c")

        self.wav_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.getcwd())
        self.bpm_value = tk.StringVar()
        self.include_beats = tk.BooleanVar(value=True)
        self.prefer_16_bar = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready.")
        self.use_anchor_finding = tk.BooleanVar(value=False)
        self.phrase_mode = tk.StringVar(value="balanced")
        self.include_bar_markers = tk.BooleanVar(value=True)
        self.include_4bar_markers = tk.BooleanVar(value=True)
        self.include_8bar_markers = tk.BooleanVar(value=True)
        self.include_16bar_markers = tk.BooleanVar(value=True)
        self.include_32bar_markers = tk.BooleanVar(value=False)
        self.include_phrase_guessing = tk.BooleanVar(value=False)
        self.preset_name = tk.StringVar(value="Programming Grid")

        self._busy = False
        self._advanced_visible = False
        self._setup_style()
        self._build_ui()
        self.include_phrase_guessing.trace_add("write", self._on_phrase_toggle)
        self.apply_preset()
        self._on_phrase_toggle()
        self._set_advanced_visible(False)

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        base_bg = "#05080c"
        panel_bg = "#0d1219"
        field_bg = "#121b24"
        edge = "#1c2b38"
        text = "#f0f7ff"
        muted = "#8da0b2"
        accent = "#11d8ff"

        style.configure(".", background=base_bg, foreground=text, fieldbackground=field_bg)
        style.configure("Panel.TFrame", background=panel_bg)
        style.configure("Panel.TLabelframe", background=panel_bg, foreground=text, bordercolor=edge, relief="solid", borderwidth=1)
        style.configure("Panel.TLabelframe.Label", background=panel_bg, foreground="#d9f6ff", font=("Segoe UI", 10, "bold"))
        style.configure("PanelTitle.TLabel", background=panel_bg, foreground="#d9f6ff", font=("Segoe UI", 10, "bold"))
        style.configure("Body.TLabel", background=panel_bg, foreground=muted, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=base_bg, foreground="#f9fdff", font=("Segoe UI", 24, "bold"))
        style.configure("SubHeader.TLabel", background=base_bg, foreground=muted, font=("Segoe UI", 10))
        style.configure("Brand.TLabel", background=base_bg, foreground=accent, font=("Segoe UI", 11, "bold"))
        style.configure("TCheckbutton", background=panel_bg, foreground=text, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", panel_bg)], foreground=[("active", text)])
        style.configure("Dark.TEntry", fieldbackground=field_bg, foreground=text, insertcolor=text, bordercolor=edge, lightcolor=edge, darkcolor=edge)
        style.configure("Dark.TCombobox", fieldbackground=field_bg, foreground=text, background=field_bg, arrowcolor=accent)
        style.map("Dark.TCombobox", fieldbackground=[("readonly", field_bg)], foreground=[("readonly", text)])
        style.configure("Neon.Horizontal.TProgressbar", troughcolor="#081019", background=accent, lightcolor=accent, darkcolor=accent, bordercolor="#081019")

    def _build_ui(self):
        base_bg = "#05080c"
        panel_bg = "#0d1219"
        field_bg = "#121b24"
        text = "#f0f7ff"
        muted = "#8da0b2"

        outer = tk.Frame(self, bg=base_bg)
        outer.pack(fill="both", expand=True, padx=18, pady=16)

        header = tk.Frame(outer, bg=base_bg)
        header.pack(fill="x", pady=(0, 14))

        brand_wrap = tk.Frame(header, bg=base_bg)
        brand_wrap.pack(side="left", fill="both", expand=True)

        ttk.Label(brand_wrap, text="STEALTH PROJECTIONS", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand_wrap, text="Laser Marker Grid Builder", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            brand_wrap,
            text="A console-style utility for BEYOND-ready marker CSV export.",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        status_wrap = tk.Frame(header, bg=base_bg)
        status_wrap.pack(side="right", anchor="ne")
        self.status_badge = tk.Label(
            status_wrap,
            textvariable=self.status_text,
            bg="#102333",
            fg="#dff8ff",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
        )
        self.status_badge.pack(anchor="e")
        self.progress = ttk.Progressbar(status_wrap, style="Neon.Horizontal.TProgressbar", mode="indeterminate", length=240)
        self.progress.pack(anchor="e", pady=(10, 0))

        columns = tk.Frame(outer, bg=base_bg)
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1, minsize=330)
        columns.grid_columnconfigure(1, weight=0, minsize=340)
        columns.grid_columnconfigure(2, weight=1, minsize=330)
        columns.grid_rowconfigure(0, weight=1)

        left = tk.Frame(columns, bg=panel_bg, width=330, highlightbackground="#1c2b38", highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_propagate(False)

        center = tk.Frame(columns, bg=panel_bg, width=340, highlightbackground="#1c2b38", highlightthickness=1)
        center.grid(row=0, column=1, sticky="ns", padx=8)
        center.grid_propagate(False)

        right = tk.Frame(columns, bg=panel_bg, width=330, highlightbackground="#1c2b38", highlightthickness=1)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right.grid_propagate(False)

        # LEFT PANEL
        self._panel_title(left, "INPUT")
        self._labeled_entry(left, "Track File", self.wav_path, browse_cmd=self.pick_wav)
        self._labeled_entry(left, "BPM Override", self.bpm_value, help_text="Optional. Leave blank to use simple auto-detect.")
        ttk.Checkbutton(left, text="Enable Anchor Finding", variable=self.use_anchor_finding).pack(anchor="w", padx=18, pady=(8, 0))
        tk.Label(
            left,
            text="When disabled, the grid starts at 0:00.000.",
            bg=panel_bg,
            fg=muted,
            font=("Segoe UI", 9),
            wraplength=280,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(2, 12))

        analyze_row = tk.Frame(left, bg=panel_bg)
        analyze_row.pack(fill="x", padx=18, pady=(6, 14))
        self.analyze_button = HoverButton(
            analyze_row,
            text="ANALYZE TRACK",
            command=self.run_analysis,
            normal_bg="#1a2430",
            hover_bg="#243443",
            fg="#e8f8ff",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=10,
        )
        self.analyze_button.pack(anchor="w")

        # CENTER PANEL
        self._panel_title(center, "MARKER OPTIONS")

        preset_wrap = tk.Frame(center, bg=panel_bg)
        preset_wrap.pack(fill="x", padx=18, pady=(4, 12))
        tk.Label(preset_wrap, text="Preset", bg=panel_bg, fg="#dff8ff", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        preset_row = tk.Frame(preset_wrap, bg=panel_bg)
        preset_row.pack(fill="x", pady=(6, 0))
        self.preset_combo = ttk.Combobox(
            preset_row,
            textvariable=self.preset_name,
            state="readonly",
            values=["Minimal", "Programming Grid", "Full Grid", "Phrase Assist"],
            width=20,
            style="Dark.TCombobox",
        )
        self.preset_combo.pack(side="left", fill="x", expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)

        self.preset_description_label = tk.Label(
            preset_wrap,
            text="",
            bg=panel_bg,
            fg="#8da0b2",
            font=("Segoe UI", 9),
            justify="left",
            anchor="w",
        )
        self.preset_description_label.pack(anchor="w", fill="x", pady=(8, 0))
        preset_wrap.bind("<Configure>", lambda e: self.preset_description_label.configure(wraplength=max(160, e.width - 10)))

        self.advanced_wrap = tk.Frame(center, bg=panel_bg, highlightbackground="#1c2b38", highlightthickness=1)
        self.advanced_wrap.pack(fill="x", padx=18, pady=(8, 14))

        adv_header = tk.Frame(self.advanced_wrap, bg=panel_bg)
        adv_header.pack(fill="x")
        tk.Label(
            adv_header,
            text="Advanced Marker Options",
            bg=panel_bg,
            fg="#dff8ff",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        ).pack(side="left", anchor="w")

        self.advanced_toggle_button = tk.Button(
            adv_header,
            text="Show",
            command=self.toggle_advanced_options,
            bg="#143042",
            fg="#dff8ff",
            activebackground="#1d4763",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            width=7,
            cursor="hand2",
        )
        self.advanced_toggle_button.pack(side="right", padx=10, pady=8)

        self.advanced_options_container = tk.Frame(self.advanced_wrap, bg=panel_bg)

        grid = tk.Frame(self.advanced_options_container, bg=panel_bg)
        grid.pack(fill="x", padx=12, pady=(4, 12))

        ttk.Checkbutton(grid, text="Beat Markers", variable=self.include_beats).grid(row=0, column=0, sticky="w", padx=(0, 18), pady=6)
        ttk.Checkbutton(grid, text="Bar Markers", variable=self.include_bar_markers).grid(row=0, column=1, sticky="w", padx=(0, 18), pady=6)
        ttk.Checkbutton(grid, text="4 Bar Markers", variable=self.include_4bar_markers).grid(row=1, column=0, sticky="w", padx=(0, 18), pady=6)
        ttk.Checkbutton(grid, text="8 Bar Markers", variable=self.include_8bar_markers).grid(row=1, column=1, sticky="w", padx=(0, 18), pady=6)
        ttk.Checkbutton(grid, text="16 Bar Markers", variable=self.include_16bar_markers).grid(row=2, column=0, sticky="w", padx=(0, 18), pady=6)
        ttk.Checkbutton(grid, text="32 Bar Markers", variable=self.include_32bar_markers).grid(row=2, column=1, sticky="w", padx=(0, 18), pady=6)
        self.phrase_guess_check = ttk.Checkbutton(grid, text="Phrase Guessing", variable=self.include_phrase_guessing)
        self.phrase_guess_check.grid(row=3, column=0, sticky="w", padx=(0, 18), pady=6)
        self.prefer_16_check = ttk.Checkbutton(grid, text="Prefer 16-Bar Phrase Resets", variable=self.prefer_16_bar)
        self.prefer_16_check.grid(row=3, column=1, sticky="w", padx=(0, 18), pady=6)

        row = tk.Frame(self.advanced_options_container, bg=panel_bg)
        row.pack(fill="x", padx=12, pady=(0, 12))
        self.phrase_sensitivity_label = ttk.Label(row, text="Phrase Sensitivity", style="PanelTitle.TLabel")
        self.phrase_sensitivity_label.pack(anchor="w")
        self.phrase_combo = ttk.Combobox(row, textvariable=self.phrase_mode, state="readonly", values=["conservative", "balanced", "aggressive"], width=18, style="Dark.TCombobox")
        self.phrase_combo.pack(anchor="w", pady=(6, 0))

        # RIGHT PANEL
        self._panel_title(right, "OUTPUT")

        self.generate_button = HoverButton(
            right,
            text="GENERATE CSV",
            command=self.generate,
            normal_bg="#00c8ff",
            hover_bg="#44e0ff",
            fg="#021018",
            font=("Segoe UI", 14, "bold"),
            padx=18,
            pady=16,
            disabled_bg="#163345",
        )
        self.generate_button.pack(fill="x", padx=18, pady=(6, 16))

        self._labeled_entry(right, "Output folder", self.output_dir, browse_cmd=self.pick_output_dir)
        open_row = tk.Frame(right, bg=panel_bg)
        open_row.pack(fill="x", padx=18, pady=(6, 16))
        self.open_button = HoverButton(
            open_row,
            text="OPEN OUTPUT FOLDER",
            command=self.open_output_dir,
            normal_bg="#143042",
            hover_bg="#1d4763",
            fg="#dff8ff",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=10,
        )
        self.open_button.pack(anchor="w")

        ttk.Label(right, text="RUN INFO", style="PanelTitle.TLabel").pack(anchor="w", padx=18, pady=(0, 8))
        text_frame = tk.Frame(right, bg="#0a0f14", highlightbackground="#192734", highlightthickness=1)
        text_frame.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.info_text = tk.Text(
            text_frame,
            height=20,
            wrap="word",
            bg="#0a0f14",
            fg=text,
            insertbackground=text,
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            padx=12,
            pady=12,
        )
        self.info_text.pack(fill="both", expand=True)
        self.info_text.insert("1.0", "No file processed yet.\n")
        self.info_text.config(state="disabled")

        footer = tk.Frame(outer, bg=base_bg)
        footer.pack(fill="x", pady=(12, 0))
        tk.Label(
            footer,
            text=f"Stealth Projections • v{VERSION}",
            bg=base_bg,
            fg=muted,
            font=("Segoe UI", 9),
        ).pack(side="left")

    def _panel_title(self, master, text):
        tk.Label(
            master,
            text=text,
            bg="#0d1219",
            fg="#dff8ff",
            font=("Segoe UI", 11, "bold"),
            padx=18,
            pady=16,
        ).pack(anchor="w")

    def _labeled_entry(self, master, label_text, variable, browse_cmd=None, help_text=None):
        panel_bg = "#0d1219"
        field_bg = "#121b24"
        text = "#f0f7ff"
        muted = "#8da0b2"

        wrap = tk.Frame(master, bg=panel_bg)
        wrap.pack(fill="x", padx=18, pady=(0, 12))

        tk.Label(wrap, text=label_text, bg=panel_bg, fg="#dff8ff", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        row = tk.Frame(wrap, bg=panel_bg)
        row.pack(fill="x", pady=(6, 0))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=0)

        entry = tk.Entry(
            row,
            textvariable=variable,
            bg=field_bg,
            fg=text,
            insertbackground=text,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#1c2b38",
            highlightcolor="#11d8ff",
            font=("Segoe UI", 10),
        )
        entry.grid(row=0, column=0, sticky="ew", ipady=10, padx=(0, 10))

        if browse_cmd:
            btn = HoverButton(
                row,
                text="BROWSE",
                command=browse_cmd,
                normal_bg="#143042",
                hover_bg="#1d4763",
                fg="#dff8ff",
                font=("Segoe UI", 9, "bold"),
                padx=14,
                pady=9,
            )
            btn.grid(row=0, column=1, sticky="e")

        if help_text:
            tk.Label(wrap, text=help_text, bg=panel_bg, fg=muted, font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))

    def _set_busy(self, busy: bool, status: str):
        self._busy = busy
        self.status_text.set(status)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

        self.analyze_button.set_enabled(not busy)
        self.generate_button.set_enabled(not busy)
        self.open_button.set_enabled(not busy)


    def _set_widget_enabled(self, widget, enabled: bool):
        try:
            widget.configure(state=("readonly" if enabled and isinstance(widget, ttk.Combobox) else ("normal" if enabled else "disabled")))
        except Exception:
            try:
                widget.state(["!disabled"] if enabled else ["disabled"])
            except Exception:
                pass

    def _on_phrase_toggle(self, *_args):
        enabled = bool(self.include_phrase_guessing.get())
        try:
            self.prefer_16_check.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            pass
        self._set_widget_enabled(self.phrase_combo, enabled)
        try:
            self.phrase_sensitivity_label.configure(foreground=("#d9f6ff" if enabled else "#5f7386"))
        except Exception:
            pass


    def apply_preset(self, _event=None):
        preset = self.preset_name.get().strip().lower()
        description = ""

        if preset == "minimal":
            self.include_beats.set(False)
            self.include_bar_markers.set(True)
            self.include_4bar_markers.set(False)
            self.include_8bar_markers.set(False)
            self.include_16bar_markers.set(True)
            self.include_32bar_markers.set(False)
            self.include_phrase_guessing.set(False)
            self.prefer_16_bar.set(False)
            self.phrase_mode.set("balanced")
            self.use_anchor_finding.set(False)
            description = "Lightweight structure view. Best for a simple timing reference."
        elif preset == "programming grid":
            self.include_beats.set(True)
            self.include_bar_markers.set(True)
            self.include_4bar_markers.set(True)
            self.include_8bar_markers.set(True)
            self.include_16bar_markers.set(True)
            self.include_32bar_markers.set(False)
            self.include_phrase_guessing.set(False)
            self.prefer_16_bar.set(False)
            self.phrase_mode.set("balanced")
            self.use_anchor_finding.set(False)
            description = "Best all-around grid for programming in BEYOND."
        elif preset == "full grid":
            self.include_beats.set(True)
            self.include_bar_markers.set(True)
            self.include_4bar_markers.set(True)
            self.include_8bar_markers.set(True)
            self.include_16bar_markers.set(True)
            self.include_32bar_markers.set(True)
            self.include_phrase_guessing.set(False)
            self.prefer_16_bar.set(False)
            self.phrase_mode.set("balanced")
            self.use_anchor_finding.set(False)
            description = "Maximum objective timing detail, including 32-bar landmarks."
        elif preset == "phrase assist":
            self.include_beats.set(True)
            self.include_bar_markers.set(True)
            self.include_4bar_markers.set(True)
            self.include_8bar_markers.set(True)
            self.include_16bar_markers.set(True)
            self.include_32bar_markers.set(False)
            self.include_phrase_guessing.set(True)
            self.prefer_16_bar.set(True)
            self.phrase_mode.set("balanced")
            self.use_anchor_finding.set(False)
            description = "Programming grid plus optional phrase suggestions for musical resets."

        self.preset_description_label.configure(text=description)
        self._on_phrase_toggle()


    def _set_advanced_visible(self, visible: bool):
        self._advanced_visible = visible
        if visible:
            self.advanced_options_container.pack(fill="x")
            self.advanced_toggle_button.configure(text="Hide")
        else:
            self.advanced_options_container.pack_forget()
            self.advanced_toggle_button.configure(text="Show")

    def toggle_advanced_options(self):
        self._set_advanced_visible(not self._advanced_visible)

    def pick_wav(self):
        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.wav_path.set(path)

    def pick_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def open_output_dir(self):
        path = self.output_dir.get().strip()
        if not path:
            return
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    def _set_info(self, text: str):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.config(state="disabled")

    def _validated_wav_path(self):
        wav_path = self.wav_path.get().strip()
        if not wav_path:
            messagebox.showerror(APP_TITLE, "Please choose a WAV file.")
            return None
        if not os.path.isfile(wav_path):
            messagebox.showerror(APP_TITLE, "Selected WAV file does not exist.")
            return None
        if not wav_path.lower().endswith(".wav"):
            messagebox.showerror(APP_TITLE, "This build currently supports WAV files only.")
            return None
        return wav_path

    def run_analysis(self):
        wav_path = self._validated_wav_path()
        if not wav_path or self._busy:
            return

        self._set_busy(True, "Analyzing...")
        thread = threading.Thread(target=self._run_analysis_worker, args=(wav_path,), daemon=True)
        thread.start()

    def _run_analysis_worker(self, wav_path):
        try:
            info = analyze_track(wav_path)
            candidates = ", ".join(str(int(c)) for c in info["candidate_bpms"])
            lines = [
                f"BPM Suggestions: {candidates if candidates else 'None found'}",
                f"Detected Anchor: {info['anchor_seconds']:.3f} s",
                f"Duration: {info['duration_seconds']:.3f} s",
                "",
                "Recommended Workflow:",
                "  If you know the BPM, type it manually.",
                "  If you do not, use the simple suggestions as a sanity check.",
                "  Turn on Anchor Finding only when the track has silence or pickup before the real start.",
            ]
            self.after(0, lambda: self._finish_success("Analysis complete.", "\n".join(lines)))
        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: self._finish_error("Analysis failed.", err))

    def generate(self):
        wav_path = self._validated_wav_path()
        if not wav_path or self._busy:
            return

        bpm_override = None
        bpm_raw = self.bpm_value.get().strip()
        if bpm_raw:
            try:
                bpm_override = clean_bpm(float(bpm_raw))
                if bpm_override <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(APP_TITLE, "BPM override must be a positive number.")
                return

        out_dir = self.output_dir.get().strip() or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        self._set_busy(True, "Generating CSV...")
        thread = threading.Thread(target=self._generate_worker, args=(wav_path, bpm_override, out_dir), daemon=True)
        thread.start()

    def _generate_worker(self, wav_path, bpm_override, out_dir):
        try:
            rows, meta = build_marker_rows(
                wav_path=wav_path,
                bpm_override=bpm_override,
                use_anchor_finding=self.use_anchor_finding.get(),
                include_beats=self.include_beats.get(),
                include_bar_markers=self.include_bar_markers.get(),
                include_4bar_markers=self.include_4bar_markers.get(),
                include_8bar_markers=self.include_8bar_markers.get(),
                include_16bar_markers=self.include_16bar_markers.get(),
                include_32bar_markers=self.include_32bar_markers.get(),
                include_phrase_guessing=self.include_phrase_guessing.get(),
                prefer_16_bar_phrases=self.prefer_16_bar.get(),
                phrase_mode=self.phrase_mode.get(),
            )
            output_name = f"{safe_stem(wav_path)}_marker_grid.csv"
            output_path = os.path.join(out_dir, output_name)
            write_csv(rows, output_path)

            bpm_display = str(int(meta["bpm_used"])) if float(meta["bpm_used"]).is_integer() else f"{meta['bpm_used']:.1f}"
            candidates = ", ".join(str(int(c)) for c in meta["candidate_bpms"])
            enabled_markers = []
            if self.include_beats.get():
                enabled_markers.append("Beat")
            if self.include_bar_markers.get():
                enabled_markers.append("Bar")
            if self.include_4bar_markers.get():
                enabled_markers.append("4 Bar")
            if self.include_8bar_markers.get():
                enabled_markers.append("8 Bar")
            if self.include_16bar_markers.get():
                enabled_markers.append("16 Bar")
            if self.include_32bar_markers.get():
                enabled_markers.append("32 Bar")
            if self.include_phrase_guessing.get():
                enabled_markers.append("Phrase")

            lines = [
                f"Output File: {output_path}",
                f"BPM Used: {bpm_display}",
                f"BPM Suggestions: {candidates}",
                f"Anchor Finding: {'On' if meta['use_anchor_finding'] else 'Off'}",
                f"Detected Anchor: {meta['detected_anchor_seconds']:.3f} s",
                f"Start Point: {meta['final_anchor_seconds']:.3f} s",
                f"Duration: {meta['duration_seconds']:.3f} s",
                f"Preset: {self.preset_name.get()}",
                f"Enabled Markers: {', '.join(enabled_markers)}",
            ]
            if self.include_phrase_guessing.get():
                lines.append(f"Phrase Sensitivity: {self.phrase_mode.get().title()}")
                lines.append(f"Phrase Bars: {meta['phrase_bars'][:30] if meta['phrase_bars'] else '[]'}")
            self.after(0, lambda: self._finish_success("CSV created.", "\n".join(lines), popup=output_path))
        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: self._finish_error("Generate failed.", err))

    def _finish_success(self, status, info_text, popup=None):
        self._set_busy(False, status)
        self._set_info(info_text)
        if popup:
            messagebox.showinfo(APP_TITLE, f"CSV created:\n{popup}")

    def _finish_error(self, status, err_text):
        self._set_busy(False, status)
        self._set_info(err_text)
        messagebox.showerror(APP_TITLE, "Operation failed.\n\nSee the run info panel for details.")


if __name__ == "__main__":
    App().mainloop()
