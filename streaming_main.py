import os
import subprocess
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
import yt_dlp
import json
import uuid
import shutil
import logging
import cv2
import base64
import threading
import time
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# TensorFlow and YAMNet imports
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    YAMNET_AVAILABLE = True
    logger.info("✅ TensorFlow and TensorFlow Hub available - YAMNet enabled")
except ImportError:
    YAMNET_AVAILABLE = False
    logger.warning("⚠️ TensorFlow/TensorFlow Hub not available - falling back to librosa analysis")

class StreamingHighlightProcessor:
    """Modified version with file output instead of direct client streaming"""
    
    def __init__(self):
        self.current_video_path = None
        self.should_stop = False  # Flag to stop processing
        
    def _greedy_peak_selection(self, peaks, scores, times, top_n, min_distance_sec):
        """
        Greedy algorithm to select peaks ensuring minimum distance between them
        """
        if len(peaks) == 0:
            return []
        
        # Sort peaks by score (descending)
        peak_scores = scores[peaks]
        sorted_indices = np.argsort(peak_scores)[::-1]  # Descending order
        sorted_peaks = peaks[sorted_indices]
        
        selected_peaks = []
        
        for peak in sorted_peaks:
            if len(selected_peaks) >= top_n:
                break
                
            peak_time = times[peak]
            
            # Check if this peak is far enough from all previously selected peaks
            too_close = False
            for selected_peak in selected_peaks:
                selected_time = times[selected_peak]
                if abs(peak_time - selected_time) < min_distance_sec:
                    too_close = True
                    break
            
            if not too_close:
                selected_peaks.append(peak)
        
        return selected_peaks

    def check_stop_condition(self, progress_tracker=None):
        """Check if processing should stop"""
        if self.should_stop:
            if progress_tracker:
                progress_tracker.sync_update("Processing stopped by user", data={"stopped": True})
            raise InterruptedError("Processing stopped by user request")
            
    def download_youtube(self, url, base_filename, progress_tracker=None, yt_format="bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]"):
        """Downloads video and audio from YouTube with progress updates"""
        video_file = base_filename + ".mp4"
        audio_file = base_filename + ".mp3"
        
        if progress_tracker:
            progress_tracker.sync_update("Starting video download...", 0.1)
        
        # Check if we should stop
        self.check_stop_condition(progress_tracker)
            
        ydl_opts_video = {
            'format': yt_format,
            'outtmpl': base_filename,
            'merge_output_format': 'mp4',
            'format_sort': ['vcodec:h264', 'vcodec:h265', 'vcodec']  # Prefer H.264, then H.265
        }
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': base_filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192'
            }]
        }
        
        try:
            print(f"Downloading video as {video_file} ...")
            # Check stop condition before starting download
            self.check_stop_condition(progress_tracker)
            
            with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                ydl.download([url])
            
            if progress_tracker:
                progress_tracker.sync_next_step("Video downloaded successfully")
            
            # Check stop condition before audio download
            self.check_stop_condition(progress_tracker)
            
            print(f"Downloading audio as {audio_file} ...")
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([url])
                
            if progress_tracker:
                progress_tracker.sync_next_step("Audio downloaded successfully")
                
        except Exception as e:
            if progress_tracker:
                progress_tracker.sync_update(f"Download error: {str(e)}", data={"error": True})
            raise
            
        return video_file, audio_file

    def detect_cheers_yamnet(self, mp3_file, progress_tracker=None, output_json='top_cheer_intervals.json', 
                            top_n=40, highlight_length=2.0, plot=False):
        """Detect cheering moments using YAMNet (Google's audio classification model)"""
        
        if not YAMNET_AVAILABLE:
            logger.warning("YAMNet not available, falling back to librosa method")
            return self.detect_cheers_librosa(mp3_file, progress_tracker, output_json, top_n, 
                                            highlight_length=highlight_length, plot=plot)
        
        if progress_tracker:
            progress_tracker.sync_update("Loading YAMNet model from TensorFlow Hub...", 0.5)
        
        try:
            # Load YAMNet model
            yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            if progress_tracker:
                progress_tracker.sync_update("YAMNet model loaded successfully", 0.52)
            
            # Load audio file
            if progress_tracker:
                progress_tracker.sync_update("Loading audio file for YAMNet analysis...", 0.55)
                
            # YAMNet expects 16kHz mono audio
            audio, sr = librosa.load(mp3_file, sr=16000, mono=True)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Audio loaded: {len(audio)} samples at {sr}Hz", 0.58)
            
            # Run YAMNet inference
            if progress_tracker:
                progress_tracker.sync_update("Running YAMNet audio classification...", 0.60)
                
            # Convert to tensor
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            
            # Get predictions
            scores, embeddings, spectrogram = yamnet_model(audio_tensor)
            
            if progress_tracker:
                progress_tracker.sync_update("YAMNet analysis complete, processing results...", 0.65)
            
            # Get class names
            class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
            class_names = []
            with tf.io.gfile.GFile(class_map_path) as csvfile:
                import csv
                reader = csv.DictReader(csvfile)
                for row in reader:
                    class_names.append(row['display_name'])
            
            # Find cheering-related classes
            cheer_classes = [
                'Cheer', 'Applause', 'Crowd', 'Shout', 'Chatter', 'Speech',
                'Roar', 'Excitement', 'Clapping', 'Whooping', 'Yell'
            ]
            
            # Find indices of cheering classes
            cheer_indices = []
            for i, class_name in enumerate(class_names):
                if any(cheer_word.lower() in class_name.lower() for cheer_word in cheer_classes):
                    cheer_indices.append(i)
                    logger.info(f"Found cheer class: {class_name} (index {i})")
            
            if not cheer_indices:
                logger.warning("No cheering classes found, using top audio energy instead")
                # Fallback to energy-based detection
                cheer_scores = np.mean(scores.numpy(), axis=1)
            else:
                # Sum scores for all cheering-related classes
                cheer_scores = np.sum(scores.numpy()[:, cheer_indices], axis=1)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Found {len(cheer_indices)} cheering-related classes", 0.68)
            
            # YAMNet produces predictions every 0.48 seconds (with some overlap)
            # Each frame represents ~0.96 seconds of audio
            frame_duration = 0.96  # seconds per YAMNet frame
            times = np.arange(len(cheer_scores)) * frame_duration
            
            if progress_tracker:
                progress_tracker.sync_update("Detecting peaks in cheer scores...", 0.70)
            
            # Find peaks in cheer scores
            # Minimum distance between peaks (in frames) - 30 seconds apart
            min_distance_sec = 60.0  # Minimum 30 seconds between highlights
            min_distance_frames = int(min_distance_sec / frame_duration)
            
            # Use adaptive threshold
            threshold = np.percentile(cheer_scores, 60)  # Top 40% of scores (lowered to get more candidates)
            peaks, properties = find_peaks(cheer_scores, 
                                         distance=min_distance_frames,
                                         height=threshold)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Found {len(peaks)} potential cheer moments (30s+ apart)", 0.72)
            
            # If we don't have enough peaks with 30s spacing, use a greedy algorithm
            if len(peaks) < top_n:
                # Try with shorter minimum distance to get more candidates
                shorter_distance_frames = int(15.0 / frame_duration)  # 15 seconds
                all_peaks, _ = find_peaks(cheer_scores, 
                                        distance=shorter_distance_frames,
                                        height=np.percentile(cheer_scores, 50))
                
                # Use greedy selection to ensure 30 second spacing
                selected_peaks = self._greedy_peak_selection(all_peaks, cheer_scores, times, 
                                                           top_n, min_distance_sec)
                
                if progress_tracker:
                    progress_tracker.sync_update(f"Used greedy selection: {len(selected_peaks)} peaks with 30s spacing", 0.74)
                
                top_peaks = sorted(selected_peaks)
            else:
                # Get top N peaks by score from the well-spaced peaks
                if len(peaks) > top_n:
                    peak_scores = cheer_scores[peaks]
                    top_peak_indices = np.argsort(peak_scores)[-top_n:]
                    top_peaks = peaks[top_peak_indices]
                    top_peaks = sorted(top_peaks)  # Sort by time
                else:
                    top_peaks = sorted(peaks)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Selected {len(top_peaks)} final cheer moments", 0.76)
            
            # Create intervals
            intervals = []
            for idx, peak in enumerate(top_peaks):
                center_time = times[peak]
                start = max(center_time - highlight_length/2, 0)
                end = min(center_time + highlight_length/2, times[-1])
                
                score = cheer_scores[peak]
                intervals.append({
                    "start": float(start),
                    "end": float(end),
                    "score": float(score),
                    "method": "yamnet"
                })
                
                if progress_tracker and (idx + 1) % 3 == 0:
                    progress_tracker.sync_update(f"Created {idx + 1}/{len(top_peaks)} YAMNet intervals", 
                                                0.78 + (idx + 1) / len(top_peaks) * 0.05)
            
            # Save results
            if progress_tracker:
                progress_tracker.sync_update("Saving YAMNet results to JSON...", 0.85)
                
            with open(output_json, 'w') as f:
                json.dump(intervals, f, indent=2)
            
            if progress_tracker:
                progress_tracker.sync_next_step(
                    f"YAMNet found {len(intervals)} cheer moments", 
                    data={"intervals": intervals, "method": "yamnet", "cheer_classes": len(cheer_indices)}
                )
            
            # Optional plotting
            if plot:
                plt.figure(figsize=(15, 8))
                
                # Plot cheer scores
                plt.subplot(2, 1, 1)
                plt.plot(times, cheer_scores, label='YAMNet Cheer Score')
                plt.scatter(times[top_peaks], cheer_scores[top_peaks], 
                           color='red', s=100, label='Top Cheer Moments', zorder=5)
                plt.axhline(y=threshold, color='orange', linestyle='--', alpha=0.7, label='Threshold')
                plt.xlabel('Time (s)')
                plt.ylabel('Cheer Score')
                plt.title('YAMNet Cheer Detection')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot spectrogram
                plt.subplot(2, 1, 2)
                plt.imshow(spectrogram.numpy().T, aspect='auto', origin='lower', 
                          extent=[0, len(audio)/sr, 0, sr/2])
                plt.ylabel('Frequency (Hz)')
                plt.xlabel('Time (s)')
                plt.title('Audio Spectrogram')
                plt.colorbar(label='Magnitude')
                
                # Mark cheer moments on spectrogram
                for peak in top_peaks:
                    peak_time = times[peak]
                    plt.axvline(x=peak_time, color='red', linestyle='-', alpha=0.8)
                
                plt.tight_layout()
                plt.show()
            
            logger.info(f"✅ YAMNet detection complete: {len(intervals)} cheer moments found")
            return intervals
            
        except Exception as e:
            logger.error(f"YAMNet analysis failed: {e}")
            if progress_tracker:
                progress_tracker.sync_update("YAMNet failed, falling back to librosa analysis...", 0.50)
            # Fallback to original method
            return self.detect_cheers_librosa(mp3_file, progress_tracker, output_json, top_n, 
                                            highlight_length=highlight_length, plot=plot)

    def detect_cheers_librosa(self, mp3_file, progress_tracker=None, output_json='top_cheer_intervals.json', 
                             top_n=40, frame_duration=0.5, hop_duration=0.1, highlight_length=2.0, plot=False):
        """Original librosa-based cheer detection (fallback method)"""
        
        if progress_tracker:
            progress_tracker.sync_update("Loading and resampling audio (librosa method)...", 0.5)
        
        # Check stop condition
        self.check_stop_condition(progress_tracker)
        
        y, sr = librosa.load(mp3_file, sr=None)
        
        if progress_tracker:
            progress_tracker.sync_update(f"Audio loaded: {len(y)} samples at {sr}Hz", 0.52)
        
        frame_length = int(sr * frame_duration)
        hop_length = int(sr * hop_duration)

        if progress_tracker:
            progress_tracker.sync_update("Computing RMS energy features...", 0.55)

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        if progress_tracker:
            progress_tracker.sync_update("Computing spectral features (STFT)...", 0.58)
            
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hi_freq_band = S[freqs > 2000, :]
        hi_freq_energy = np.mean(hi_freq_band, axis=0)
        
        if progress_tracker:
            progress_tracker.sync_update("Normalizing audio features...", 0.62)
            
        rms_norm = rms / np.max(rms)
        hi_freq_norm = hi_freq_energy / np.max(hi_freq_energy)
        cheer_score = rms_norm * hi_freq_norm
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        if progress_tracker:
            progress_tracker.sync_update("Detecting peaks in cheer score...", 0.65)

        # Use 30 second minimum distance for consistency with YAMNet
        min_distance_sec = 30.0  # Minimum 30 seconds between highlights
        min_distance = int(min_distance_sec / hop_duration)
        peaks, _ = find_peaks(cheer_score, distance=min_distance)
        
        if progress_tracker:
            progress_tracker.sync_update(f"Found {len(peaks)} potential peaks (30s+ apart)", 0.68)
        
        # If we don't have enough peaks with 30s spacing, use greedy selection
        if len(peaks) < top_n:
            # Try with shorter minimum distance to get more candidates
            shorter_distance = int(15.0 / hop_duration)  # 15 seconds
            all_peaks, _ = find_peaks(cheer_score, distance=shorter_distance)
            
            # Use greedy selection to ensure 30 second spacing
            selected_peaks = self._greedy_peak_selection(all_peaks, cheer_score, times, 
                                                       top_n, min_distance_sec)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Used greedy selection: {len(selected_peaks)} peaks with 30s spacing", 0.69)
            
            top_peaks = sorted(selected_peaks)
        else:
            # Get top N peaks by score from the well-spaced peaks
            if len(peaks) > top_n:
                top_peaks = peaks[np.argsort(cheer_score[peaks])[-top_n:]]
                if progress_tracker:
                    progress_tracker.sync_update(f"Selected top {top_n} peaks from {len(peaks)} candidates", 0.70)
            else:
                top_peaks = peaks
                if progress_tracker:
                    progress_tracker.sync_update(f"Using all {len(peaks)} peaks (less than {top_n})", 0.70)
                
        top_peaks = sorted(top_peaks)

        if progress_tracker:
            progress_tracker.sync_update("Creating highlight intervals...", 0.72)

        intervals = []
        for idx, peak in enumerate(top_peaks):
            center = times[peak]
            start = max(center - highlight_length/2, 0)
            end = min(center + highlight_length/2, times[-1])
            intervals.append({
                "start": float(start), 
                "end": float(end),
                "score": float(cheer_score[peak]),
                "method": "librosa"
            })
            
            if progress_tracker and (idx + 1) % 3 == 0:  # Update every 3 intervals
                progress_tracker.sync_update(f"Created {idx + 1}/{len(top_peaks)} highlight intervals", 0.72 + (idx + 1) / len(top_peaks) * 0.05)
        
        if progress_tracker:
            progress_tracker.sync_update("Saving intervals to JSON...", 0.78)
            
        with open(output_json, 'w') as f:
            json.dump(intervals, f, indent=2)
        
        if progress_tracker:
            progress_tracker.sync_next_step(
                f"Found {len(intervals)} highlight moments (librosa)", 
                data={"intervals": intervals, "method": "librosa"}
            )

        if plot:
            plt.figure(figsize=(12,6))
            plt.plot(times, cheer_score, label='Cheer Score (RMS × High-freq)')
            plt.scatter(times[top_peaks], cheer_score[top_peaks], color='red', label='Top Cheers')
            plt.xlabel('Time (s)')
            plt.ylabel('Score')
            plt.title('Top Cheering Moments (Librosa)')
            plt.legend()
            plt.tight_layout()
            plt.show()

        return intervals

    def detect_cheers(self, mp3_file, progress_tracker=None, output_json='top_cheer_intervals.json', 
                     top_n=40, frame_duration=0.5, hop_duration=0.1, highlight_length=2.0, plot=False):
        """Detect cheering moments - tries YAMNet first, falls back to librosa"""
        
        if YAMNET_AVAILABLE:
            logger.info("🎯 Using YAMNet for audio classification-based cheer detection")
            return self.detect_cheers_yamnet(mp3_file, progress_tracker, output_json, top_n, highlight_length, plot)
        else:
            logger.info("🎵 Using librosa for energy-based cheer detection")
            return self.detect_cheers_librosa(mp3_file, progress_tracker, output_json, top_n, 
                                            frame_duration, hop_duration, highlight_length, plot)

    def make_highlight_reel(self, video_path, mp3_audio, intervals_json, progress_tracker=None, 
                           video_streamer=None, final_output='tmp/highlight_reel.mp4',
                           output_dir='clips', sr_target=44100, trim_secs=600, padding=12.0,
                           min_clip_length=2.0, fps=60, fade_duration=1.0, watermark_path='watermark.png'):
        """Create highlight reel with progress updates, live video streaming, and watermark"""
        
        # Ensure tmp directory exists
        os.makedirs("tmp", exist_ok=True)
        
        # Check if watermark exists
        has_watermark = os.path.exists(watermark_path)
        if has_watermark:
            logger.info(f"✅ Watermark found: {watermark_path}")
        else:
            logger.info(f"⚠️ Watermark not found: {watermark_path} - proceeding without watermark")
        
        # Set video for streaming
        if video_streamer:
            video_streamer.set_video_path(video_path)
            video_streamer.start_streaming()
        
        temp_video_audio = f"{uuid.uuid4().hex}_video_audio.wav"
        temp_stretched_mp3 = f"{uuid.uuid4().hex}_stretched.mp3"
        temp_synced_mp3 = f"{uuid.uuid4().hex}_stretched_synced.mp3"
        temp_merged_video = f"{uuid.uuid4().hex}_synced_video.mp4"

        os.makedirs(output_dir, exist_ok=True)

        def get_duration(path):
            cmd = [
                'ffprobe', '-i', path, '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ]
            out = subprocess.check_output(cmd)
            return float(out.strip())

        if progress_tracker:
            progress_tracker.sync_update("Analyzing video and audio durations...", 0.8)

        video_dur = get_duration(video_path)
        mp3_dur = get_duration(mp3_audio)
        print(f"Video duration: {video_dur:.3f}s, MP3 duration: {mp3_dur:.3f}s")

        if progress_tracker:
            progress_tracker.sync_update(f"Video: {video_dur:.1f}s, Audio: {mp3_dur:.1f}s", 0.82)

        # Stretch MP3 to match video duration
        speed = video_dur / mp3_dur
        print(f"Stretching MP3 by atempo={speed:.8f}")
        
        if progress_tracker:
            progress_tracker.sync_update(f"Syncing audio (speed: {speed:.3f}x)...", 0.85)
        
        if 0.5 <= speed <= 2.0:
            tempo_str = f"atempo={speed}"
        else:
            s1 = speed**0.5
            s2 = speed / s1
            tempo_str = f"atempo={s1},atempo={s2}"

        cmd = [
            "ffmpeg", "-y", "-i", mp3_audio, "-filter:a", tempo_str, temp_stretched_mp3
        ]
        subprocess.run(cmd, check=True)
        print(f"Stretched MP3 saved as: {temp_stretched_mp3}")

        if progress_tracker:
            progress_tracker.sync_update("Extracting audio from video for sync...", 0.87)
            
        if not os.path.exists(temp_video_audio):
            print("Extracting audio from video...")
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr_target), temp_video_audio
            ]
            subprocess.run(cmd, check=True)
            if progress_tracker:
                progress_tracker.sync_update("Audio extraction complete", 0.875)
        else:
            print("Audio already extracted from video.")

        # Cross-correlate to find offset
        if progress_tracker:
            progress_tracker.sync_update("Loading audio files for cross-correlation...", 0.88)
            
        print("Loading audio for cross-correlation sync...")
        y1, sr1 = librosa.load(temp_stretched_mp3, sr=sr_target)
        y2, sr2 = librosa.load(temp_video_audio, sr=sr_target)
        
        if progress_tracker:
            progress_tracker.sync_update("Trimming audio samples for analysis...", 0.885)
            
        samples = int(sr_target * trim_secs)
        y1 = y1[:samples]
        y2 = y2[:samples]
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]
        
        if progress_tracker:
            progress_tracker.sync_update("Computing FFT-based cross-correlation...", 0.89)
            
        print("Calculating FFT-based cross-correlation to find offset...")
        corr = correlate(y1, y2, mode='full', method='fft')
        lags = np.arange(-min_len + 1, min_len)
        best_lag = lags[np.argmax(corr)]
        offset_sec = best_lag / sr_target
        print(f"\n>>> Detected offset: {offset_sec:.2f} seconds")

        if progress_tracker:
            progress_tracker.sync_update("Applying audio synchronization...", 0.91)

        print("Syncing MP3 audio to video start...")
        if abs(offset_sec) < 0.05:
            synced_mp3_use = temp_stretched_mp3
            print("No extra sync needed, audio is already aligned.")
            if progress_tracker:
                progress_tracker.sync_update("Audio already synchronized", 0.915)
        else:
            if offset_sec > 0:
                offset_ms = int(offset_sec * 1000)
                print(f"Delaying MP3 by {offset_ms} ms ...")
                if progress_tracker:
                    progress_tracker.sync_update(f"Adding {offset_ms}ms delay to audio...", 0.912)
                cmd = [
                    "ffmpeg", "-y", "-i", temp_stretched_mp3, "-af",
                    f"adelay={offset_ms}|{offset_ms}", temp_synced_mp3
                ]
                subprocess.run(cmd, check=True)
            elif offset_sec < 0:
                abs_offset = abs(offset_sec)
                print(f"Trimming {abs_offset:.2f} seconds from the start of MP3 ...")
                if progress_tracker:
                    progress_tracker.sync_update(f"Trimming {abs_offset:.2f}s from audio start...", 0.912)
                cmd = [
                    "ffmpeg", "-y", "-ss", str(abs_offset), "-i", temp_stretched_mp3,
                    "-acodec", "copy", temp_synced_mp3
                ]
                subprocess.run(cmd, check=True)
            synced_mp3_use = temp_synced_mp3

        if progress_tracker:
            progress_tracker.sync_update("Merging synchronized audio with video...", 0.92)

        print("Merging video and fully synced MP3 (for highlight extraction)...")
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", synced_mp3_use,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-shortest", temp_merged_video
        ]
        subprocess.run(cmd, check=True)
        print(f"Final synced video for highlight extraction: {temp_merged_video}")
        
        if progress_tracker:
            progress_tracker.sync_update("Video merge complete, preparing to extract clips...", 0.925)

        # Update video for streaming
        if video_streamer:
            video_streamer.set_video_path(temp_merged_video)

        def frame_align(t, fps):
            return round(t * fps) / fps

        with open(intervals_json, "r") as f:
            intervals = json.load(f)

        if progress_tracker:
            progress_tracker.sync_update(f"Extracting {len(intervals)} highlight clips...", 0.93)

        clip_paths = []
        video_duration = float(get_duration(video_path))
        
        for idx, seg in enumerate(intervals):
            start = max(frame_align(seg["start"] - padding, fps), 0)
            end = min(frame_align(seg["end"] + padding, fps), video_duration)
            duration = end - start
            if duration < min_clip_length:
                if progress_tracker:
                    progress_tracker.sync_update(f"Skipping clip {idx+1} (too short: {duration:.2f}s)", 0.93 + (idx / len(intervals)) * 0.04)
                continue
                
            random_name = f"{uuid.uuid4().hex}.mp4"
            outclip = os.path.join(output_dir, random_name)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Extracting clip {idx+1}/{len(intervals)}: {start:.1f}s-{end:.1f}s", 0.93 + (idx / len(intervals)) * 0.04)
                
            # Extract clip with watermark if available
            if has_watermark:
                # Add watermark during clip extraction (bottom-right, 50x50px, 50% opacity, no padding)
                cmd = [
                    "ffmpeg", "-y", "-ss", str(start), "-i", temp_merged_video,
                    "-i", watermark_path,
                    "-filter_complex", 
                    "[1:v]scale=50:50,format=rgba,colorchannelmixer=aa=0.5[watermark];[0:v][watermark]overlay=W-w:H-h",
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac", "-strict", "-2",
                    outclip
                ]
            else:
                # Extract clip without watermark
                cmd = [
                    "ffmpeg", "-y", "-ss", str(start), "-i", temp_merged_video,
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac", "-strict", "-2",
                    outclip
                ]
                
            print(f"Extracting clip {idx+1}: {start:.3f}s to {end:.3f}s -> {outclip}" + 
                  (" (with watermark)" if has_watermark else ""))
            subprocess.run(cmd, check=True)
            clip_paths.append(outclip)

        def get_clip_duration(path):
            cmd = [
                'ffprobe', '-i', path, '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ]
            out = subprocess.check_output(cmd)
            return float(out.strip())

        if progress_tracker:
            progress_tracker.sync_update("Analyzing clip durations for crossfade...", 0.975)

        clip_durations = [get_clip_duration(clip) for clip in clip_paths]

        num_clips = len(clip_paths)
        if num_clips < 2:
            if progress_tracker:
                progress_tracker.sync_update("Single clip detected, copying to output...", 0.98)
            print("Not enough clips for crossfade. Copying single clip to output.")
            subprocess.run(["cp", clip_paths[0], final_output])
        else:
            if progress_tracker:
                progress_tracker.sync_update(f"Creating crossfade filter for {num_clips} clips...", 0.98)

            input_cmds = []
            for clip in clip_paths:
                input_cmds += ["-i", clip]

            if progress_tracker:
                progress_tracker.sync_update("Building FFmpeg filter complex...", 0.985)

            v_prev = "[0:v]"
            a_prev = "[0:a]"
            filter_parts = []
            cum_duration = 0.0
            for i in range(1, num_clips):
                dur_prev = clip_durations[i-1]
                offset = cum_duration + dur_prev - fade_duration * i
                v_next = f"[{i}:v]"
                a_next = f"[{i}:a]"
                v_out = f"[v{i}]"
                a_out = f"[a{i}]"
                filter_parts.append(
                    f"{v_prev}{v_next}xfade=transition=fade:duration={fade_duration}:offset={offset}{v_out};"
                )
                filter_parts.append(
                    f"{a_prev}{a_next}acrossfade=d={fade_duration}:c1=tri:c2=tri{a_out};"
                )
                v_prev = v_out
                a_prev = a_out
                cum_duration += dur_prev

            filter_complex = "".join(filter_parts)
            if filter_complex.endswith(";"):
                filter_complex = filter_complex[:-1]

            if progress_tracker:
                progress_tracker.sync_update("Rendering final crossfade video...", 0.99)

            cmd = ["ffmpeg", "-y"] + input_cmds + [
                "-filter_complex", filter_complex,
                "-map", v_prev, "-map", a_prev,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-strict", "-2",
                final_output
            ]

            print("Running FFmpeg for crossfade highlight reel...")
            subprocess.run(cmd, check=True)

        # Update final video for streaming
        if video_streamer:
            video_streamer.set_video_path(final_output)

        print(f"\n🎉 Done! Your highlight reel (with fade transitions" + 
              (" and watermark" if has_watermark else "") + 
              f") is: {os.path.abspath(final_output)}")

        if progress_tracker:
            progress_tracker.sync_next_step(
                "Highlight reel completed! File ready for download." + 
                (" (Watermark applied)" if has_watermark else ""), 
                data={"output_path": final_output, "clip_count": len(clip_paths), "watermarked": has_watermark}
            )

        print("Deleting temporary highlight clips and temp files...")
        for f in clip_paths + [temp_video_audio, temp_stretched_mp3, temp_synced_mp3, temp_merged_video]:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🗑️  Deleted {f}")
            except Exception as e:
                print(f"Warning: could not delete {f}: {e}")

        try:
            if len(os.listdir(output_dir)) == 0:
                shutil.rmtree(output_dir)
        except Exception:
            pass

        print("All temp files cleaned up!")
        return final_output
        """Create highlight reel with progress updates and live video streaming"""
        
        # Ensure tmp directory exists
        os.makedirs("tmp", exist_ok=True)
        
        # Set video for streaming
        if video_streamer:
            video_streamer.set_video_path(video_path)
            video_streamer.start_streaming()
        
        temp_video_audio = f"{uuid.uuid4().hex}_video_audio.wav"
        temp_stretched_mp3 = f"{uuid.uuid4().hex}_stretched.mp3"
        temp_synced_mp3 = f"{uuid.uuid4().hex}_stretched_synced.mp3"
        temp_merged_video = f"{uuid.uuid4().hex}_synced_video.mp4"

        os.makedirs(output_dir, exist_ok=True)

        def get_duration(path):
            cmd = [
                'ffprobe', '-i', path, '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ]
            out = subprocess.check_output(cmd)
            return float(out.strip())

        if progress_tracker:
            progress_tracker.sync_update("Analyzing video and audio durations...", 0.8)

        video_dur = get_duration(video_path)
        mp3_dur = get_duration(mp3_audio)
        print(f"Video duration: {video_dur:.3f}s, MP3 duration: {mp3_dur:.3f}s")

        if progress_tracker:
            progress_tracker.sync_update(f"Video: {video_dur:.1f}s, Audio: {mp3_dur:.1f}s", 0.82)

        # Stretch MP3 to match video duration
        speed = video_dur / mp3_dur
        print(f"Stretching MP3 by atempo={speed:.8f}")
        
        if progress_tracker:
            progress_tracker.sync_update(f"Syncing audio (speed: {speed:.3f}x)...", 0.85)
        
        if 0.5 <= speed <= 2.0:
            tempo_str = f"atempo={speed}"
        else:
            s1 = speed**0.5
            s2 = speed / s1
            tempo_str = f"atempo={s1},atempo={s2}"

        cmd = [
            "ffmpeg", "-y", "-i", mp3_audio, "-filter:a", tempo_str, temp_stretched_mp3
        ]
        subprocess.run(cmd, check=True)
        print(f"Stretched MP3 saved as: {temp_stretched_mp3}")

        if progress_tracker:
            progress_tracker.sync_update("Extracting audio from video for sync...", 0.87)
            
        if not os.path.exists(temp_video_audio):
            print("Extracting audio from video...")
            cmd = [
                "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr_target), temp_video_audio
            ]
            subprocess.run(cmd, check=True)
            if progress_tracker:
                progress_tracker.sync_update("Audio extraction complete", 0.875)
        else:
            print("Audio already extracted from video.")

        # Cross-correlate to find offset
        if progress_tracker:
            progress_tracker.sync_update("Loading audio files for cross-correlation...", 0.88)
            
        print("Loading audio for cross-correlation sync...")
        y1, sr1 = librosa.load(temp_stretched_mp3, sr=sr_target)
        y2, sr2 = librosa.load(temp_video_audio, sr=sr_target)
        
        if progress_tracker:
            progress_tracker.sync_update("Trimming audio samples for analysis...", 0.885)
            
        samples = int(sr_target * trim_secs)
        y1 = y1[:samples]
        y2 = y2[:samples]
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]
        
        if progress_tracker:
            progress_tracker.sync_update("Computing FFT-based cross-correlation...", 0.89)
            
        print("Calculating FFT-based cross-correlation to find offset...")
        corr = correlate(y1, y2, mode='full', method='fft')
        lags = np.arange(-min_len + 1, min_len)
        best_lag = lags[np.argmax(corr)]
        offset_sec = best_lag / sr_target
        print(f"\n>>> Detected offset: {offset_sec:.2f} seconds")

        if progress_tracker:
            progress_tracker.sync_update("Applying audio synchronization...", 0.91)

        print("Syncing MP3 audio to video start...")
        if abs(offset_sec) < 0.05:
            synced_mp3_use = temp_stretched_mp3
            print("No extra sync needed, audio is already aligned.")
            if progress_tracker:
                progress_tracker.sync_update("Audio already synchronized", 0.915)
        else:
            if offset_sec > 0:
                offset_ms = int(offset_sec * 1000)
                print(f"Delaying MP3 by {offset_ms} ms ...")
                if progress_tracker:
                    progress_tracker.sync_update(f"Adding {offset_ms}ms delay to audio...", 0.912)
                cmd = [
                    "ffmpeg", "-y", "-i", temp_stretched_mp3, "-af",
                    f"adelay={offset_ms}|{offset_ms}", temp_synced_mp3
                ]
                subprocess.run(cmd, check=True)
            elif offset_sec < 0:
                abs_offset = abs(offset_sec)
                print(f"Trimming {abs_offset:.2f} seconds from the start of MP3 ...")
                if progress_tracker:
                    progress_tracker.sync_update(f"Trimming {abs_offset:.2f}s from audio start...", 0.912)
                cmd = [
                    "ffmpeg", "-y", "-ss", str(abs_offset), "-i", temp_stretched_mp3,
                    "-acodec", "copy", temp_synced_mp3
                ]
                subprocess.run(cmd, check=True)
            synced_mp3_use = temp_synced_mp3

        if progress_tracker:
            progress_tracker.sync_update("Merging synchronized audio with video...", 0.92)

        print("Merging video and fully synced MP3 (for highlight extraction)...")
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", synced_mp3_use,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-shortest", temp_merged_video
        ]
        subprocess.run(cmd, check=True)
        print(f"Final synced video for highlight extraction: {temp_merged_video}")
        
        if progress_tracker:
            progress_tracker.sync_update("Video merge complete, preparing to extract clips...", 0.925)

        # Update video for streaming
        if video_streamer:
            video_streamer.set_video_path(temp_merged_video)

        def frame_align(t, fps):
            return round(t * fps) / fps

        with open(intervals_json, "r") as f:
            intervals = json.load(f)

        if progress_tracker:
            progress_tracker.sync_update(f"Extracting {len(intervals)} highlight clips...", 0.93)

        clip_paths = []
        video_duration = float(get_duration(video_path))
        
        for idx, seg in enumerate(intervals):
            start = max(frame_align(seg["start"] - padding, fps), 0)
            end = min(frame_align(seg["end"] + padding, fps), video_duration)
            duration = end - start
            if duration < min_clip_length:
                if progress_tracker:
                    progress_tracker.sync_update(f"Skipping clip {idx+1} (too short: {duration:.2f}s)", 0.93 + (idx / len(intervals)) * 0.04)
                continue
                
            random_name = f"{uuid.uuid4().hex}.mp4"
            outclip = os.path.join(output_dir, random_name)
            
            if progress_tracker:
                progress_tracker.sync_update(f"Extracting clip {idx+1}/{len(intervals)}: {start:.1f}s-{end:.1f}s", 0.93 + (idx / len(intervals)) * 0.04)
                
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", temp_merged_video,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-strict", "-2",
                outclip
            ]
            print(f"Extracting clip {idx+1}: {start:.3f}s to {end:.3f}s -> {outclip}")
            subprocess.run(cmd, check=True)
            clip_paths.append(outclip)

        def get_clip_duration(path):
            cmd = [
                'ffprobe', '-i', path, '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ]
            out = subprocess.check_output(cmd)
            return float(out.strip())

        if progress_tracker:
            progress_tracker.sync_update("Analyzing clip durations for crossfade...", 0.975)

        clip_durations = [get_clip_duration(clip) for clip in clip_paths]

        num_clips = len(clip_paths)
        if num_clips < 2:
            if progress_tracker:
                progress_tracker.sync_update("Single clip detected, copying to output...", 0.98)
            print("Not enough clips for crossfade. Copying single clip to output.")
            subprocess.run(["cp", clip_paths[0], final_output])
        else:
            if progress_tracker:
                progress_tracker.sync_update(f"Creating crossfade filter for {num_clips} clips...", 0.98)

            input_cmds = []
            for clip in clip_paths:
                input_cmds += ["-i", clip]

            if progress_tracker:
                progress_tracker.sync_update("Building FFmpeg filter complex...", 0.985)

            v_prev = "[0:v]"
            a_prev = "[0:a]"
            filter_parts = []
            cum_duration = 0.0
            for i in range(1, num_clips):
                dur_prev = clip_durations[i-1]
                offset = cum_duration + dur_prev - fade_duration * i
                v_next = f"[{i}:v]"
                a_next = f"[{i}:a]"
                v_out = f"[v{i}]"
                a_out = f"[a{i}]"
                filter_parts.append(
                    f"{v_prev}{v_next}xfade=transition=fade:duration={fade_duration}:offset={offset}{v_out};"
                )
                filter_parts.append(
                    f"{a_prev}{a_next}acrossfade=d={fade_duration}:c1=tri:c2=tri{a_out};"
                )
                v_prev = v_out
                a_prev = a_out
                cum_duration += dur_prev

            filter_complex = "".join(filter_parts)
            if filter_complex.endswith(";"):
                filter_complex = filter_complex[:-1]

            if progress_tracker:
                progress_tracker.sync_update("Rendering final crossfade video...", 0.99)

            cmd = ["ffmpeg", "-y"] + input_cmds + [
                "-filter_complex", filter_complex,
                "-map", v_prev, "-map", a_prev,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac", "-strict", "-2",
                final_output
            ]

            print("Running FFmpeg for crossfade highlight reel...")
            subprocess.run(cmd, check=True)

        # Update final video for streaming
        if video_streamer:
            video_streamer.set_video_path(final_output)

        print(f"\n🎉 Done! Your highlight reel (with fade transitions) is: {os.path.abspath(final_output)}")

        if progress_tracker:
            progress_tracker.sync_next_step(
                "Highlight reel completed! File ready for download.", 
                data={"output_path": final_output, "clip_count": len(clip_paths)}
            )

        print("Deleting temporary highlight clips and temp files...")
        for f in clip_paths + [temp_video_audio, temp_stretched_mp3, temp_synced_mp3, temp_merged_video]:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🗑️  Deleted {f}")
            except Exception as e:
                print(f"Warning: could not delete {f}: {e}")

        try:
            if len(os.listdir(output_dir)) == 0:
                shutil.rmtree(output_dir)
        except Exception:
            pass

        print("All temp files cleaned up!")
        return final_output

    def main_with_streaming(self, youtube_url, progress_tracker=None, video_streamer=None, 
                           fade_duration=1.0, padding=12.0, fps=60, 
                           yt_format='bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]'):
        """Main function with streaming capabilities - RETURNS file path instead of sending to client"""
        # Generate a random base name WITHOUT extension
        base_filename = uuid.uuid4().hex
        clips_dir = f"clips_{uuid.uuid4().hex[:8]}"
        intervals_json = f"{uuid.uuid4().hex}.json"
        
        # Create tmp directory and generate unique output filename
        os.makedirs("tmp", exist_ok=True)
        final_output = f"tmp/{uuid.uuid4().hex}_highlight.mp4"
        
        # Initialize variables to avoid scope issues
        video_file = None
        audio_file = None

        try:
            if progress_tracker:
                progress_tracker.sync_update("Starting YouTube highlight processing...", 0.0)

            # Download (yt-dlp will add extensions)
            video_file, audio_file = self.download_youtube(
                youtube_url, base_filename, progress_tracker, yt_format=yt_format
            )
            
            # Detect cheer intervals
            self.detect_cheers(audio_file, progress_tracker, intervals_json)
            
            # Make highlight reel - saves to tmp directory
            result_path = self.make_highlight_reel(
                video_path=video_file,
                mp3_audio=audio_file,
                intervals_json=intervals_json,
                progress_tracker=progress_tracker,
                video_streamer=video_streamer,
                final_output=final_output,
                output_dir=clips_dir,
                padding=padding,
                fps=fps,
                fade_duration=fade_duration
            )
            
            # Return the path to the saved file (don't delete it yet)
            print(f"✅ Final video saved to: {result_path}")
            return result_path
            
        except Exception as e:
            if progress_tracker:
                progress_tracker.sync_update(f"Error: {str(e)}", data={"error": True})
            raise
        finally:
            # Clean up only temporary files, NOT the final output
            cleanup_files = []
            if video_file and video_file != final_output:
                cleanup_files.append(video_file)
            if audio_file and audio_file != final_output:
                cleanup_files.append(audio_file)
            cleanup_files.append(intervals_json)
            
            for f in cleanup_files:
                try:
                    if f and os.path.exists(f):
                        os.remove(f)
                        print(f"🗑️  Deleted temp file: {f}")
                except Exception as e:
                    print(f"Warning: could not delete {f}: {e}")

            # Clean up clips directory
            try:
                if os.path.exists(clips_dir):
                    shutil.rmtree(clips_dir)
                    print(f"🗑️  Deleted clips directory: {clips_dir}")
            except Exception:
                pass
            
            print("✅ Cleanup complete (final video preserved in tmp)")

    # Original main function for backward compatibility
    def main(self, youtube_url, keep_final_output=True, fade_duration=1.0, padding=12.0, 
             fps=60, yt_format='bestvideo[vcodec!*=av01][height<=720]+bestaudio/best[height<=720]'):
        """Original main function for backward compatibility"""
        return self.main_with_streaming(
            youtube_url=youtube_url,
            progress_tracker=None,
            video_streamer=None,
            fade_duration=fade_duration,
            padding=padding,
            fps=fps,
            yt_format=yt_format
        )

# WebSocket video streaming class (disabled for stability)
class WebSocketVideoStreamer:
    def __init__(self, session_id: str, connection_manager):
        self.session_id = session_id
        self.manager = connection_manager
        self.current_video_path = None
        self.streaming = False
        self.stream_thread = None
        
    def set_video_path(self, video_path: str):
        """Set the current video file to stream - disabled for stability"""
        self.current_video_path = video_path
        logger.info(f"Video path updated for session {self.session_id}: {os.path.basename(video_path)}")
        
    def start_streaming(self):
        """Start streaming video frames - disabled for stability"""
        logger.info(f"Live video streaming disabled for session {self.session_id} (file-based download available)")
        # Note: Live streaming is disabled to prevent WebSocket threading issues
        # The final video will be available for download when processing completes
    
    def stop_streaming(self):
        """Stop streaming video frames - no-op since streaming is disabled"""
        logger.info(f"Video streaming stopped for session {self.session_id}")

# Synchronous progress tracker wrapper
class SyncProgressTracker:
    """Wrapper to handle async progress updates from sync code"""
    def __init__(self, async_tracker):
        self.async_tracker = async_tracker
        
    def sync_update(self, message: str, progress: float = None, data: dict = None):
        """Send progress update synchronously"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.async_tracker.send_update(message, progress, data)
            )
            loop.close()
        except Exception as e:
            logger.error(f"Progress update error: {e}")
    
    def sync_next_step(self, message: str, data: dict = None):
        """Send next step update synchronously"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.async_tracker.next_step(message, data)
            )
            loop.close()
        except Exception as e:
            logger.error(f"Progress step error: {e}")