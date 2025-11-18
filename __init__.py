"""
Anki Add-on: Audio Recording with Transcription Interface
Automatically records audio when reviewing cards and provides a React-based
interface for transcription using Whisper Web.
Enhanced with streaming file writing for growing audio files.
"""

from aqt import mw, gui_hooks, tr
from aqt.qt import *
from aqt.utils import disable_help_button, showInfo
from aqt.sound import QtAudioInputRecorder
from aqt.sound import NativeMacRecorder
from anki.utils import is_mac
from typing import Callable, Optional
import os
import time
import base64
import json
import wave
import struct

try:
    from PyQt6.QtWebChannel import QWebChannel
    from PyQt6.QtWebEngineCore import (
        QWebEngineProfile, 
        QWebEnginePage, 
        QWebEngineSettings,
        QWebEngineUrlRequestInterceptor
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
    WEBCHANNEL_AVAILABLE = True
except ImportError:
    WEBCHANNEL_AVAILABLE = False
    print("[Add-on] PyQt6 WebEngine not available")

# Add-on directories
ADDON_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(ADDON_DIR, "audio")
WEB_DIR = os.path.join(ADDON_DIR, "")

# Create directories
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Global variables
_current_rec_dialog: Optional["RecordDialog"] = None
_audio_bridge: Optional["AudioBridge"] = None
react_window = None


def _addon_audio_path(card_id: Optional[int] = None) -> str:
    """Generate audio file path for recording"""
    if card_id is not None:
        return os.path.join(AUDIO_DIR, f"{card_id}_rec.wav")
    else:
        ts = int(time.time() * 1000)
        return os.path.join(AUDIO_DIR, f"{ts}_rec.wav")


class StreamingWaveWriter:
    """Write WAV file incrementally as data comes in"""
    
    def __init__(self, filepath: str, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frames_written = 0
        self.wav_file = None
        self._open_file()
    
    def _open_file(self):
        """Open WAV file and write header"""
        self.wav_file = wave.open(self.filepath, 'wb')
        self.wav_file.setnchannels(self.channels)
        self.wav_file.setsampwidth(self.sample_width)
        self.wav_file.setframerate(self.sample_rate)
    
    def write_frames(self, data: bytes):
        """Write audio frames to file"""
        if self.wav_file:
            self.wav_file.writeframes(data)
            self.frames_written += len(data) // (self.channels * self.sample_width)
    
    def close(self):
        """Close the WAV file"""
        if self.wav_file:
            self.wav_file.close()
            self.wav_file = None
    
    def get_duration(self) -> float:
        """Get current duration in seconds"""
        if self.sample_rate > 0:
            return self.frames_written / self.sample_rate
        return 0.0


class StreamingAudioRecorder(QObject):
    """Custom audio recorder that writes to file as it records"""
    
    def __init__(self, output_path: str, parent=None):
        super().__init__(parent)
        self.output_path = output_path
        self.audio_source = None
        self.io_device = None
        self.wave_writer = None
        self.is_recording = False
        self.start_time = None
        self.timer = None
        
        # Audio format settings
        self.format = QAudioFormat()
        self.format.setSampleRate(16000)
        self.format.setChannelCount(1)
        self.format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        print(f"[StreamingAudioRecorder] Initialized for {output_path}")
    
    def start(self, callback: Optional[Callable] = None):
        """Start recording"""
        print(f"[StreamingAudioRecorder] start() called")
        try:
            # Get default audio input device
            audio_device = QMediaDevices.defaultAudioInput()
            print(f"[StreamingAudioRecorder] Audio device: {audio_device}")
            
            if not audio_device or audio_device.isNull():
                print("[StreamingAudioRecorder] ERROR: No audio input device found")
                if callback:
                    callback()
                return
            
            print(f"[StreamingAudioRecorder] Audio device description: {audio_device.description()}")
            
            # Check if format is supported, get actual format used
            if not audio_device.isFormatSupported(self.format):
                print("[StreamingAudioRecorder] WARNING: Format not supported, finding nearest")
                # Get the nearest supported format
                nearest = audio_device.preferredFormat()
                print(f"[StreamingAudioRecorder] Device preferred format: rate={nearest.sampleRate()}, channels={nearest.channelCount()}, format={nearest.sampleFormat()}")
            
            # Create audio source
            self.audio_source = QAudioSource(audio_device, self.format, self)
            
            # Set volume to maximum
            self.audio_source.setVolume(1.0)
            
            # Set buffer size (important for getting readyRead signals)
            self.audio_source.setBufferSize(8192)  # Larger buffer
            print(f"[StreamingAudioRecorder] Audio source created with buffer size: {self.audio_source.bufferSize()}")
            
            # Initialize wave writer - use actual format from audio source
            actual_format = self.audio_source.format()
            self.wave_writer = StreamingWaveWriter(
                self.output_path,
                sample_rate=actual_format.sampleRate(),
                channels=actual_format.channelCount(),
                sample_width=2  # 16-bit = 2 bytes
            )
            print(f"[StreamingAudioRecorder] Wave writer created with rate={actual_format.sampleRate()}, channels={actual_format.channelCount()}")
            
            # Start recording
            self.io_device = self.audio_source.start()
            print(f"[StreamingAudioRecorder] IO device: {self.io_device}")
            
            if self.io_device:
                self.io_device.readyRead.connect(self._on_ready_read)
                self.is_recording = True
                self.start_time = time.time()
                
                # Check initial state
                print(f"[StreamingAudioRecorder] Initial state: {self.audio_source.state()}")
                print(f"[StreamingAudioRecorder] Initial error: {self.audio_source.error()}")
                print(f"[StreamingAudioRecorder] Volume: {self.audio_source.volume()}")
                print(f"[StreamingAudioRecorder] Format sample rate: {self.audio_source.format().sampleRate()}")
                print(f"[StreamingAudioRecorder] Recording started successfully to {self.output_path}")
                
                # Start a timer to poll for data and check state
                self.timer = QTimer(self)
                self.timer.timeout.connect(self._poll_data)
                self.timer.start(50)  # Poll every 50ms
                print(f"[StreamingAudioRecorder] Started polling timer")
            else:
                print("[StreamingAudioRecorder] ERROR: Failed to get IO device")
            
            if callback:
                callback()
                
        except Exception as e:
            print(f"[StreamingAudioRecorder] EXCEPTION in start(): {e}")
            import traceback
            traceback.print_exc()
            if callback:
                callback()
    
    def _poll_data(self):
        """Poll for data periodically (backup if readyRead doesn't fire)"""
        if not self.io_device or not self.wave_writer or not self.is_recording:
            return
        
        try:
            # Check audio source state
            state = self.audio_source.state()
            error = self.audio_source.error()
            bytes_available = self.io_device.bytesAvailable()
            
            # Log state occasionally (every 20 polls = 1 second)
            if not hasattr(self, '_poll_count'):
                self._poll_count = 0
            self._poll_count += 1
            
            if self._poll_count % 20 == 0:
                print(f"[StreamingAudioRecorder] State={state}, Error={error}, BytesAvailable={bytes_available}")
            
            if bytes_available > 0:
                print(f"[StreamingAudioRecorder] Polling: {bytes_available} bytes available")
                self._read_and_write_data()
        except Exception as e:
            print(f"[StreamingAudioRecorder] EXCEPTION in _poll_data: {e}")
    
    def _on_ready_read(self):
        """Called when audio data is available"""
        print(f"[StreamingAudioRecorder] _on_ready_read signal fired!")
        self._read_and_write_data()
    
    def _read_and_write_data(self):
        """Read audio data and write to file"""
        if not self.io_device or not self.wave_writer:
            return
        
        try:
            # Check how much data is available
            bytes_available = self.io_device.bytesAvailable()
            print(f"[StreamingAudioRecorder] Attempting to read {bytes_available} bytes")
            
            # Read available data
            data = self.io_device.readAll()
            data_len = len(data) if data else 0
            
            if data_len > 0:
                # print(f"[StreamingAudioRecorder] Writing {data_len} bytes of audio data")
                # Write to WAV file
                self.wave_writer.write_frames(bytes(data))
                
                # Send to audio bridge if available
                global _audio_bridge
                if _audio_bridge:
                    _audio_bridge.sendAudioChunk(bytes(data))
            else:
                print(f"[StreamingAudioRecorder] WARNING: No data read (bytesAvailable was {bytes_available})")
                    
        except Exception as e:
            print(f"[StreamingAudioRecorder] EXCEPTION reading data: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self, callback: Optional[Callable[[str], None]] = None):
        """Stop recording"""
        print(f"[StreamingAudioRecorder] stop() called")
        try:
            self.is_recording = False
            
            # Stop polling timer
            if self.timer:
                self.timer.stop()
                self.timer = None
            
            # Read any remaining data
            if self.io_device and self.wave_writer:
                print(f"[StreamingAudioRecorder] Reading any remaining data")
                bytes_available = self.io_device.bytesAvailable()
                print(f"[StreamingAudioRecorder] Bytes available before stop: {bytes_available}")
                if bytes_available > 0:
                    self._read_and_write_data()
            
            if self.audio_source:
                print(f"[StreamingAudioRecorder] Stopping audio source (state={self.audio_source.state()})")
                self.audio_source.stop()
            
            if self.wave_writer:
                print(f"[StreamingAudioRecorder] Closing wave writer (frames written: {self.wave_writer.frames_written})")
                self.wave_writer.close()
            
            file_size = os.path.getsize(self.output_path) if os.path.exists(self.output_path) else 0
            print(f"[StreamingAudioRecorder] Stopped recording, saved to {self.output_path} ({file_size} bytes)")
            
            if callback:
                callback(self.output_path)
                
        except Exception as e:
            print(f"[StreamingAudioRecorder] EXCEPTION in stop(): {e}")
            import traceback
            traceback.print_exc()
            if callback:
                callback(self.output_path)
    
    def duration(self) -> float:
        """Get current recording duration"""
        if self.wave_writer:
            return self.wave_writer.get_duration()
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def on_timer(self):
        """Called periodically to update status"""
        pass  # Duration is handled by wave_writer


class AudioBridge(QObject):
    """Bridge class to handle communication between Anki and web AudioRecorder"""
    
    def __init__(self, web_view):
        super().__init__()
        self.web_view = web_view
        self.is_recording = False
        self.audio_chunks = []
    
    @pyqtSlot(str, result=str)
    def startRecording(self, data: str) -> str:
        """Called from JS to start recording"""
        try:
            self.is_recording = True
            self.audio_chunks = []
            _start_record_on_question()
            return '{"status": "started"}'
        except Exception as e:
            return f'{{"status": "error", "message": "{str(e)}"}}'
    
    @pyqtSlot(str, result=str)
    def stopRecording(self, data: str) -> str:
        """Called from JS to stop recording"""
        try:
            self.is_recording = False
            _stop_record_on_answer()
            return '{"status": "stopped"}'
        except Exception as e:
            return f'{{"status": "error", "message": "{str(e)}"}}'
    
    @pyqtSlot(str, result=str)
    def getRecordingStatus(self, data: str) -> str:
        """Called from JS to check recording status"""
        return f'{{"recording": {"true" if self.is_recording else "false"}}}'
    
    def sendAudioChunk(self, chunk_data: bytes):
        """Send audio chunk to web interface"""
        if not self.is_recording:
            return
        
        try:
            encoded = base64.b64encode(chunk_data).decode('utf-8')
            js = f"window.ankiAudioBridge && window.ankiAudioBridge.receiveAudioChunk('{encoded}');"
            self._eval_js(js)
        except Exception as e:
            print(f"Error sending audio chunk: {e}")
    
    def notifyRecordingComplete(self, file_path: str):
        """Notify web interface that recording is complete"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    audio_data = f.read()
                
                encoded = base64.b64encode(audio_data).decode('utf-8')
                js = f"window.ankiAudioBridge && window.ankiAudioBridge.receiveCompleteAudio('{encoded}');"
                self._eval_js(js)
        except Exception as e:
            print(f"Error sending complete audio: {e}")
    
    def _eval_js(self, js: str):
        """Execute JavaScript in the web view"""
        try:
            if hasattr(self.web_view, 'eval'):
                self.web_view.eval(js)
            elif hasattr(self.web_view, 'runJavaScript'):
                self.web_view.runJavaScript(js)
        except Exception as e:
            print(f"JS eval error: {e}")


class FileApiHandler(QObject):
    """Handle file API requests from JavaScript for saving transcripts"""
    
    @pyqtSlot(str, str, result=str)
    def saveTranscriptToCard(self, card_id_str, transcript):
        """Save transcript to an Anki card field"""
        try:
            card_id = int(card_id_str)
            print(f"[FileApi] Saving transcript to card ID: {card_id}")
            print(f"[FileApi] Transcript: {transcript}")
            
            card = mw.col.get_card(card_id)
            note = card.note()
            
            if "Transcript" in note:
                note["Transcript"] = transcript
                mw.col.update_note(note)
                
                # Refresh the card display
                if mw.reviewer and mw.reviewer.card:
                    mw.reviewer.card.load()
                    mw.reviewer._redraw_current_card()
                
                return json.dumps({
                    'success': True,
                    'message': 'Transcript saved to card'
                })
            else:
                return json.dumps({
                    'success': False,
                    'error': 'Card does not have a Transcript field. Please add one to your note type.'
                })
                
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })


class RecordDialog(QDialog):
    """Minimal recording dialog that starts immediately and saves to an explicit path"""
    
    def __init__(self, parent, mw_obj, on_success: Callable[[str], None], card_id: Optional[int] = None):
        super().__init__(parent)
        self.mw = mw_obj
        self._on_success = on_success
        disable_help_button(self)
        self.setWindowTitle("Recording")
        self.label = QLabel("Recording...")
        
        h = QHBoxLayout()
        h.addWidget(QLabel())
        h.addWidget(self.label)
        v = QVBoxLayout()
        v.addLayout(h)
        self.setLayout(v)

        self._out = _addon_audio_path(card_id)
        
        # Use native recorders (they handle permissions better)
        # Try NativeMacRecorder on Mac, then QtAudioInputRecorder
        self._recorder = None
        
        if is_mac and NativeMacRecorder:
            try:
                self._recorder = NativeMacRecorder(self._out)
                print("[RecordDialog] Using NativeMacRecorder")
                # Patch it to write incrementally if possible
                self._patch_native_mac_recorder()
            except Exception as e:
                print(f"[RecordDialog] NativeMacRecorder failed: {e}")
        
        if not self._recorder and QtAudioInputRecorder:
            try:
                self._recorder = QtAudioInputRecorder(self._out, self.mw, self)
                print("[RecordDialog] Using QtAudioInputRecorder")
            except Exception as e:
                print(f"[RecordDialog] QtAudioInputRecorder failed: {e}")
        
        if not self._recorder:
            print("[RecordDialog] ERROR: No recorder available")

        # Add a timer to monitor file size growth
        self._size_timer = QTimer(self)
        self._size_timer.timeout.connect(self._check_file_size)
        self._last_size = 0

        try:
            if self._recorder:
                self._recorder.start(self._start_timer)
            else:
                self._start_timer()
        except Exception as e:
            print(f"[RecordDialog] Error starting recorder: {e}")
            self._start_timer()
        
        self.show()
    
    def _patch_native_mac_recorder(self):
        """Attempt to make NativeMacRecorder write incrementally"""
        try:
            # NativeMacRecorder uses AVFoundation which writes to file as it records
            # It should already grow the file, so just monitor it
            print("[RecordDialog] NativeMacRecorder should write incrementally by default")
        except Exception as e:
            print(f"[RecordDialog] Failed to patch recorder: {e}")
    
    def _check_file_size(self):
        """Monitor file size to verify recording is working"""
        if os.path.exists(self._out):
            current_size = os.path.getsize(self._out)
            if current_size != self._last_size:
                # print(f"[RecordDialog] File size: {current_size} bytes (was {self._last_size})")
                self._last_size = current_size
            elif current_size == self._last_size and self._last_size > 0:
                # File stopped growing - might be an issue
                pass

    def _start_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(200)
        
        # Start file size monitoring timer
        self._size_timer.start(500)  # Check every 500ms

    def _on_timer(self) -> None:
        if not getattr(self, "_recorder", None):
            return
        
        try:
            self._recorder.on_timer()
            dur = self._recorder.duration()
            self.label.setText(tr.media_recordingtime(secs=f"{dur:0.1f}") if tr else f"Recording: {dur:0.1f}s")
            
            # Send duration update to web interface
            global _audio_bridge
            if _audio_bridge:
                js = f"window.ankiAudioBridge && window.ankiAudioBridge.updateDuration({dur});"
                _audio_bridge._eval_js(js)
        except Exception as e:
            print(f"[RecordDialog] Timer error: {e}")
    
    def accept(self) -> None:
        if getattr(self, "_timer", None):
            self._timer.stop()
        if getattr(self, "_recorder", None):
            try:
                self._recorder.stop(self._on_success)
            except Exception:
                self._on_success(getattr(self, "_out", ""))
        else:
            self._on_success(getattr(self, "_out", ""))
        super().accept()

    def reject(self) -> None:
        if getattr(self, "_timer", None):
            self._timer.stop()
        
        def _cleanup(path: str) -> None:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        
        if getattr(self, "_recorder", None):
            try:
                self._recorder.stop(_cleanup)
            except Exception:
                _cleanup(getattr(self, "_out", ""))
        else:
            _cleanup(getattr(self, "_out", ""))
        
        super().reject()


class RequestInterceptor(QWebEngineUrlRequestInterceptor):
    """Intercept requests to add proper headers for WASM and CORS"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def interceptRequest(self, info):
        """Add headers to requests"""
        info.setHttpHeader(b"Cross-Origin-Embedder-Policy", b"require-corp")
        info.setHttpHeader(b"Cross-Origin-Opener-Policy", b"same-origin")
        info.setHttpHeader(b"Cross-Origin-Resource-Policy", b"cross-origin")


class ReactWorkerWindow(QDialog):
    """Window to display the React transcription interface"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper Web Transcription")
        self.resize(500, 700)
        
        main_layout = QVBoxLayout()
        
        if not WEBCHANNEL_AVAILABLE:
            label = QLabel("PyQt6 WebEngine not available. Please install PyQt6-WebEngine.")
            main_layout.addWidget(label)
            self.setLayout(main_layout)
            return
        
        # Create profile with interceptor
        self.profile = QWebEngineProfile("addon_profile", self)
        self.interceptor = RequestInterceptor(self)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        
        # Create page with custom profile
        self.page = QWebEnginePage(self.profile, self)
        
        # Connect to console message handler
        self.page.javaScriptConsoleMessage = self.handle_console_message
        
        # Create web view
        self.web = QWebEngineView()
        self.web.setPage(self.page)
        
        # Enable necessary settings
        settings = self.web.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        
        # Create file API bridge
        self.file_api = FileApiHandler()
        self.channel = QWebChannel(self.page)
        self.channel.registerObject("fileApi", self.file_api)
        self.page.setWebChannel(self.channel)
        print("[Add-on] WebChannel initialized successfully")
        
        # Connect load events
        self.web.page().loadFinished.connect(self.on_load_finished)
        self.web.page().loadFinished.connect(self.inject_helpers)
        
        main_layout.addWidget(self.web)
        self.setLayout(main_layout)
        
        # Load the app
        index_path = os.path.join(WEB_DIR, "index.html")
        file_url = QUrl.fromLocalFile(index_path)
        print(f"[WebView] Loading: {file_url.toString()}")
        self.web.setUrl(file_url)
    
    def handle_console_message(self, level, message, line_number, source_id):
        """Handle JavaScript console messages"""
        level_str = {
            0: "INFO",
            1: "WARNING", 
            2: "ERROR"
        }.get(level, "UNKNOWN")
        
        print(f"[JS Console {level_str}] {message} (line {line_number})")
        if source_id:
            print(f"[JS Console] Source: {source_id}")
    
    def on_load_finished(self, ok):
        if ok:
            print("[WebView] Page loaded successfully")
        else:
            print("[WebView] Page load failed")
    
    def inject_helpers(self, ok):
        """Inject helper JavaScript for file operations and audio bridge"""
        if not ok:
            return
        
        js_code = """
        // Load QWebChannel script
        var script = document.createElement('script');
        script.src = 'qrc:///qtwebchannel/qwebchannel.js';
        script.onload = function() {
            console.log('[WebChannel] qwebchannel.js loaded');
            
            if (typeof qt !== 'undefined' && qt.webChannelTransport) {
                new QWebChannel(qt.webChannelTransport, function(channel) {
                    window.fileApi = channel.objects.fileApi;
                    console.log('[WebChannel] fileApi available');
                    window.dispatchEvent(new Event('fileApiReady'));
                });
            }
        };
        document.head.appendChild(script);
        
        // Audio bridge for recording integration
        window.ankiAudioBridge = {
            receiveAudioChunk: function(base64Data) {
                if (window.onAnkiAudioChunk) {
                    window.onAnkiAudioChunk(base64Data);
                }
            },
            receiveCompleteAudio: function(base64Data) {
                if (window.onAnkiAudioComplete) {
                    window.onAnkiAudioComplete(base64Data);
                }
            },
            updateDuration: function(seconds) {
                if (window.onAnkiDurationUpdate) {
                    window.onAnkiDurationUpdate(seconds);
                }
            }
        };
        
        console.log('[Add-on] Audio bridge and helpers initialized');
        """
        
        self.web.page().runJavaScript(js_code)


def _on_recording_saved(path: str) -> None:
    """Called when recording is saved"""
    print(f"[Recording] Saved to: {path}")
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        print(f"[Recording] File size: {file_size} bytes")
    
    global _audio_bridge
    if _audio_bridge:
        _audio_bridge.notifyRecordingComplete(path)


def _start_record_on_question(*_args, **_kwargs) -> None:
    """Start recording when question is shown"""
    print("[_start_record_on_question] Called")
    global _current_rec_dialog
    parent = getattr(mw, "dialog", mw)
    
    # Get current card ID
    card_id = None
    try:
        if hasattr(mw, "reviewer") and mw.reviewer and hasattr(mw.reviewer, "card"):
            card = mw.reviewer.card
            if card:
                card_id = card.id
                print(f"[_start_record_on_question] Card ID: {card_id}")
    except Exception as e:
        print(f"[_start_record_on_question] Error getting card ID: {e}")
    
    if _current_rec_dialog is None:
        try:
            print(f"[_start_record_on_question] Creating RecordDialog")
            _current_rec_dialog = RecordDialog(parent, mw, _on_recording_saved, card_id)
            print(f"[_start_record_on_question] RecordDialog created successfully")
        except Exception as e:
            print(f"[_start_record_on_question] EXCEPTION: Failed to start: {e}")
            import traceback
            traceback.print_exc()
            _current_rec_dialog = None
    else:
        print(f"[_start_record_on_question] Recording dialog already exists")

    if react_window and card and WEBCHANNEL_AVAILABLE:
        card_id = card.id
        
        # Get the full audio file path
        audio_file_path = _addon_audio_path(card_id)
        
        # Convert to file:// URL for proper loading
        audio_url = QUrl.fromLocalFile(audio_file_path).toString()
        
        # Update JavaScript with current card info
        js_code = f"""
            window.currentCardAudio = '{audio_url}';
            window.currentCardId = '{card_id}';
            window.currentCardTranscript = '';
            console.log('Card ID:', window.currentCardId);
            console.log('Audio URL:', window.currentCardAudio);
            window.dispatchEvent(new CustomEvent('cardChanged', {{ 
                detail: {{ 
                    cardId: {card_id},
                    audioUrl: '{audio_url}'
                }} 
            }}));
        """
        
        react_window.web.page().runJavaScript(js_code)
    
    show_react_app()


def _stop_record_on_answer(*_args, **_kwargs) -> None:
    """Stop recording when answer is shown"""
    global _current_rec_dialog
    if _current_rec_dialog is None:
        return
    
    try:
        _current_rec_dialog.accept()
    except Exception:
        cb = getattr(_current_rec_dialog, "_on_success", _on_recording_saved)
        try:
            cb(getattr(_current_rec_dialog, "_out", ""))
        except Exception:
            pass
    
    try:
        _current_rec_dialog.close()
    except Exception:
        pass
    
    _current_rec_dialog = None


def _inject_js_on_question(*_args, **_kwargs) -> None:
    """Inject JavaScript bridge when question is shown"""
    global _audio_bridge
    
    try:
        view = None
        if getattr(mw, "reviewer", None) and getattr(mw.reviewer, "web", None):
            view = mw.reviewer.web
        elif getattr(mw, "web", None):
            view = mw.web
        
        if not view:
            return
        
        # Initialize audio bridge
        if not _audio_bridge:
            _audio_bridge = AudioBridge(view)
        
        # Inject JS
        js = """
        console.log('awesomestt: audio bridge injected');
        
        window.ankiAudioBridge = {
            receiveAudioChunk: function(base64Data) {
                if (window.onAnkiAudioChunk) {
                    window.onAnkiAudioChunk(base64Data);
                }
            },
            receiveCompleteAudio: function(base64Data) {
                if (window.onAnkiAudioComplete) {
                    window.onAnkiAudioComplete(base64Data);
                }
            },
            updateDuration: function(seconds) {
                if (window.onAnkiDurationUpdate) {
                    window.onAnkiDurationUpdate(seconds);
                }
            }
        };
        """
        
        if hasattr(view, "eval"):
            view.eval(js)
        elif hasattr(view, "runJavaScript"):
            view.runJavaScript(js)
    except Exception as e:
        print(f"[JS Injection] Error: {e}")


def show_react_app():
    """Show the React transcription window"""
    global react_window
    if react_window and react_window.isVisible():
        react_window.raise_()
        react_window.activateWindow()
    elif react_window:
        react_window.show()
        react_window.move(100, 100)
        react_window.raise_()
        react_window.activateWindow()


def on_show_answer(card):
    """Handle answer display - update card info and show transcription window"""
    global react_window
    
    if react_window and card and WEBCHANNEL_AVAILABLE:
        card_id = card.id
        
        # Get the full audio file path
        audio_file_path = _addon_audio_path(card_id)
        
        # Convert to file:// URL for proper loading
        audio_url = QUrl.fromLocalFile(audio_file_path).toString()
        
        print(f"[on_show_answer] Card ID: {card_id}")
        print(f"[on_show_answer] Audio file path: {audio_file_path}")
        print(f"[on_show_answer] Audio URL: {audio_url}")
        
        # Update JavaScript with current card info
        js_code = f"""
            window.currentCardAudio = '{audio_url}';
            window.currentCardId = '{card_id}';
            window.currentCardTranscript = '';
            console.log('Card ID:', window.currentCardId);
            console.log('Audio URL:', window.currentCardAudio);
            window.dispatchEvent(new CustomEvent('cardChanged', {{ 
                detail: {{ 
                    cardId: {card_id},
                    audioUrl: '{audio_url}'
                }} 
            }}));
        """
        
        print(f"[on_show_answer] Executing JS code")
        react_window.web.page().runJavaScript(js_code)
    
    show_react_app()


def init_react_window():
    """Initialize the React transcription window"""
    global react_window
    react_window = ReactWorkerWindow(mw)
    react_window.show()
    react_window.move(100, 100)
    print("[Add-on] React transcription window initialized")


def setup_addon():
    """Initialize the add-on"""
    print("[Add-on] Setting up merged audio recording + transcription add-on")
    
    # Initialize React window
    init_react_window()
    
    # Hook into Anki events
    gui_hooks.reviewer_did_show_question.append(_start_record_on_question)
    gui_hooks.reviewer_did_show_question.append(_inject_js_on_question)
    gui_hooks.reviewer_did_show_answer.append(_stop_record_on_answer)
    gui_hooks.reviewer_did_show_answer.append(on_show_answer)
    
    print("[Add-on] Setup complete")


# Initialize when Anki loads
gui_hooks.main_window_did_init.append(setup_addon)