"""Small synchronous, headless libretro frontend. One core per OS process.

ABI reference: https://github.com/libretro/RetroArch/blob/master/libretro-common/include/libretro.h
No emulation, input automation, ROM patching, or reward logic lives here.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np


class GameInfo(C.Structure):
    _fields_ = [("path", C.c_char_p), ("data", C.c_void_p),
                ("size", C.c_size_t), ("meta", C.c_char_p)]


class SystemInfo(C.Structure):
    _fields_ = [("library_name", C.c_char_p), ("library_version", C.c_char_p),
                ("valid_extensions", C.c_char_p), ("need_fullpath", C.c_bool),
                ("block_extract", C.c_bool)]


class Geometry(C.Structure):
    _fields_ = [("base_width", C.c_uint), ("base_height", C.c_uint),
                ("max_width", C.c_uint), ("max_height", C.c_uint), ("aspect_ratio", C.c_float)]


class Timing(C.Structure):
    _fields_ = [("fps", C.c_double), ("sample_rate", C.c_double)]


class AVInfo(C.Structure):
    _fields_ = [("geometry", Geometry), ("timing", Timing)]


class Variable(C.Structure):
    _fields_ = [("key", C.c_char_p), ("value", C.c_char_p)]


class InputDescriptor(C.Structure):
    _fields_ = [("port", C.c_uint), ("device", C.c_uint), ("index", C.c_uint),
                ("id", C.c_uint), ("description", C.c_char_p)]


ENV = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIO_BATCH = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
POLL = C.CFUNCTYPE(None)
INPUT = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)
LOG = C.CFUNCTYPE(None, C.c_int, C.c_char_p)


class LogInterface(C.Structure):
    _fields_ = [("log", LOG)]


class LibretroHost:
    """Owns one native core. Separate processes are mandatory for parallel games."""
    _active_pid = None
    _used_pid = None

    def __init__(self, core_path: Path, rom_path: Path, root: Path | None = None, *, require_ram=True, audio=False):
        if self.__class__._used_pid == os.getpid():
            raise RuntimeError("Only one native core lifetime is supported per process; reset it or spawn a new worker.")
        self.root = Path(root or Path(__file__).resolve().parent)
        self.core_path, self.rom_path = Path(core_path).resolve(), Path(rom_path).resolve()
        for path in (self.core_path, self.rom_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.closed = False
        self.loaded = False
        self.rotation = 0
        self.pixel_format = 0
        self.buttons: set[int] = set()
        self.frame = np.zeros((260, 224, 3), dtype=np.uint8)
        self.capture = True
        self.callback_error = None
        self.shutdown_requested = False
        self.variables: dict[str, bytes] = {}
        self.descriptors = []
        self.messages = []
        self.audio_enabled = bool(audio)
        self._audio_chunks: list[np.ndarray] = []
        self._directory_buffers = {}
        for name in ("system", "saves"):
            directory = self.root / "runtime" / (f"saves-{os.getpid()}" if name == "saves" else name)
            directory.mkdir(parents=True, exist_ok=True)
            # Some older native core paths copy a whole MAX_PATH buffer.
            self._directory_buffers[name] = C.create_string_buffer(str(directory).encode("utf-8"), 4096)
        audio_batch = self._audio_batch if self.audio_enabled else (lambda data, n: n)
        self._callbacks = (ENV(self._environment), VIDEO(self._video),
                           AUDIO(lambda l, r: None), AUDIO_BATCH(audio_batch),
                           POLL(lambda: None), INPUT(self._input))
        # The libretro logger is variadic. Ignore its format arguments entirely;
        # on the Windows x64 ABI extra arguments do not affect this callback.
        self._log_callback = LOG(lambda level, fmt: None)
        self.dll = C.CDLL(str(self.core_path))
        self._bind()
        for setter, cb in zip(("environment", "video_refresh", "audio_sample", "audio_sample_batch", "input_poll", "input_state"), self._callbacks):
            getattr(self.dll, "retro_set_" + setter)(cb)
        self.dll.retro_init()
        self.__class__._active_pid = os.getpid()
        self.__class__._used_pid = os.getpid()
        try:
            self.system_info = SystemInfo()
            self.dll.retro_get_system_info(C.byref(self.system_info))
            self._rom_string = str(self.rom_path).encode("utf-8")
            info = GameInfo(self._rom_string, None, 0, None)
            if not self.dll.retro_load_game(C.byref(info)):
                raise RuntimeError("FBNeo could not load ROM: " + str(self.rom_path) + "; " + "; ".join(self.messages[-5:]))
            self.loaded = True
            self.dll.retro_set_controller_port_device(0, 1)
            self.av_info = AVInfo()
            self.dll.retro_get_system_av_info(C.byref(self.av_info))
            self.fps = float(self.av_info.timing.fps)
            self.sample_rate = float(self.av_info.timing.sample_rate) or 44100.0
            ptr = self.dll.retro_get_memory_data(2)
            size = int(self.dll.retro_get_memory_size(2))
            if require_ram and (not ptr or size < 0x1000):
                raise RuntimeError(f"FBNeo did not expose expected Pac-Man system RAM (got {size} bytes)")
            self.ram = np.ctypeslib.as_array((C.c_ubyte * size).from_address(ptr)) if ptr and size else None
        except Exception:
            self.close()
            raise

    def _bind(self):
        specs = {
            "retro_init": (None, []), "retro_deinit": (None, []),
            "retro_reset": (None, []), "retro_run": (None, []),
            "retro_unload_game": (None, []),
            "retro_get_system_info": (None, [C.POINTER(SystemInfo)]),
            "retro_get_system_av_info": (None, [C.POINTER(AVInfo)]),
            "retro_load_game": (C.c_bool, [C.POINTER(GameInfo)]),
            "retro_set_controller_port_device": (None, [C.c_uint, C.c_uint]),
            "retro_get_memory_data": (C.c_void_p, [C.c_uint]),
            "retro_get_memory_size": (C.c_size_t, [C.c_uint]),
            "retro_serialize_size": (C.c_size_t, []),
            "retro_serialize": (C.c_bool, [C.c_void_p, C.c_size_t]),
            "retro_unserialize": (C.c_bool, [C.c_void_p, C.c_size_t]),
        }
        for name, cb in zip(("environment", "video_refresh", "audio_sample", "audio_sample_batch", "input_poll", "input_state"), (ENV, VIDEO, AUDIO, AUDIO_BATCH, POLL, INPUT)):
            specs["retro_set_" + name] = (None, [cb])
        for name, (restype, argtypes) in specs.items():
            fn = getattr(self.dll, name)
            fn.restype, fn.argtypes = restype, argtypes

    def _environment(self, command, data):
        try:
            cmd = command & 0xFFFF
            if cmd == 1:  # SET_ROTATION, counterclockwise quarter turns
                self.rotation = C.cast(data, C.POINTER(C.c_uint))[0]
                return True
            if cmd == 3:  # GET_CAN_DUPE
                C.cast(data, C.POINTER(C.c_bool))[0] = True
                return True
            if cmd == 6:  # SET_MESSAGE (first field is char*)
                value = C.cast(data, C.POINTER(C.c_char_p))[0]
                if value:
                    self.messages.append(value.decode("utf-8", "replace"))
                return True
            if cmd == 7:
                self.shutdown_requested = True
                return True
            if cmd in (9, 31):
                buf = self._directory_buffers["system" if cmd == 9 else "saves"]
                C.cast(data, C.POINTER(C.c_void_p))[0] = C.addressof(buf)
                return True
            if cmd == 10:  # SET_PIXEL_FORMAT
                fmt = C.cast(data, C.POINTER(C.c_int))[0]
                if fmt not in (0, 1, 2):
                    return False
                self.pixel_format = fmt
                return True
            if cmd == 11:  # SET_INPUT_DESCRIPTORS
                descriptors = C.cast(data, C.POINTER(InputDescriptor))
                self.descriptors = []
                for index in range(1024):
                    item = descriptors[index]
                    if not item.description:
                        break
                    self.descriptors.append({"port": item.port, "device": item.device,
                                             "id": item.id, "description": item.description.decode("utf-8", "replace")})
                return True
            if cmd == 15:  # GET_VARIABLE
                var = C.cast(data, C.POINTER(Variable)).contents
                var.value = self.variables.get(var.key.decode("utf-8"))
                return var.value is not None
            if cmd == 16:  # SET_VARIABLES, legacy fallback requested by version 0
                variables = C.cast(data, C.POINTER(Variable))
                for index in range(4096):
                    var = variables[index]
                    if not var.key:
                        break
                    key = var.key.decode("utf-8")
                    options = var.value.decode("utf-8").split(";", 1)[-1].strip().split("|")
                    value = options[0]
                    # Disable extras and persistent high-score injection, never game rules.
                    for token in ("hiscore", "cpu-overclock", "frameskip"):
                        if token in key:
                            for candidate in ("disabled", "100%", "0"):
                                if candidate in options:
                                    value = candidate
                                    break
                    self.variables.setdefault(key, value.encode("utf-8"))
                return True
            if cmd == 17:  # GET_VARIABLE_UPDATE
                C.cast(data, C.POINTER(C.c_bool))[0] = False
                return True
            if cmd == 27:  # GET_LOG_INTERFACE
                C.cast(data, C.POINTER(LogInterface)).contents.log = self._log_callback
                return True
            if cmd in (18, 32, 34, 35, 36, 37, 42, 44, 52, 55):
                return True
            if cmd in (39, 59, 60):  # language English, core-options v0, message v0
                C.cast(data, C.POINTER(C.c_uint))[0] = 0
                return True
            if cmd == 47:  # GET_AUDIO_VIDEO_ENABLE (bit 0 video, bit 1 audio)
                C.cast(data, C.POINTER(C.c_int))[0] = 3 if self.audio_enabled else 1
                return True
            if cmd == 49:  # GET_FASTFORWARDING
                C.cast(data, C.POINTER(C.c_bool))[0] = True
                return True
            if cmd == 72:  # GET_SAVESTATE_CONTEXT
                if data:
                    C.cast(data, C.POINTER(C.c_uint))[0] = 0
                return True
            return False
        except Exception as exc:
            self.callback_error = exc
            return False

    def _input(self, port, device, index, button):
        if port != 0 or device != 1:
            return 0
        if button == 256:
            return sum(1 << b for b in self.buttons)
        return int(button in self.buttons)

    def _video(self, data, width, height, pitch):
        if not data or not self.capture:
            return
        try:
            bpp = 4 if self.pixel_format == 1 else 2
            dtype = np.uint32 if bpp == 4 else np.uint16
            raw = np.frombuffer(C.string_at(data, pitch * height), dtype=dtype).reshape(height, pitch // bpp)[:, :width]
            if self.pixel_format == 1:
                rgb = np.stack(((raw >> 16) & 255, (raw >> 8) & 255, raw & 255), axis=-1).astype(np.uint8)
            elif self.pixel_format == 2:
                rgb = np.stack((((raw >> 11) & 31) * 255 // 31, ((raw >> 5) & 63) * 255 // 63,
                                (raw & 31) * 255 // 31), axis=-1).astype(np.uint8)
            else:
                rgb = np.stack((((raw >> 10) & 31) * 255 // 31, ((raw >> 5) & 31) * 255 // 31,
                                (raw & 31) * 255 // 31), axis=-1).astype(np.uint8)
            self.frame = np.ascontiguousarray(np.rot90(rgb, self.rotation))
        except Exception as exc:
            self.callback_error = exc

    def _audio_batch(self, data, frames):
        # Interleaved stereo int16, frames * 2 samples. Copy immediately:
        # the core owns this buffer and reuses it.
        try:
            if data and frames:
                self._audio_chunks.append(np.frombuffer(C.string_at(data, frames * 4), dtype=np.int16).copy())
        except Exception as exc:
            self.callback_error = exc
        return frames

    def drain_audio(self) -> np.ndarray:
        """Return buffered stereo int16 samples and clear the buffer."""
        if not self._audio_chunks:
            return np.empty(0, dtype=np.int16)
        samples = np.concatenate(self._audio_chunks)
        self._audio_chunks.clear()
        return samples

    def run(self, buttons=(), *, capture=True):
        self.buttons = set(buttons)
        self.capture = capture
        self.dll.retro_run()
        if self.callback_error:
            raise RuntimeError("libretro callback failed") from self.callback_error
        if self.shutdown_requested:
            raise RuntimeError("libretro core requested shutdown")
        return self.frame

    def reset(self):
        self.buttons.clear()
        self.dll.retro_reset()

    def save_state(self) -> bytes:
        size = self.dll.retro_serialize_size()
        state = C.create_string_buffer(size)
        if not self.dll.retro_serialize(state, size):
            raise RuntimeError("libretro save-state failed")
        return state.raw

    def load_state(self, state: bytes):
        buffer = C.create_string_buffer(state)
        if not self.dll.retro_unserialize(buffer, len(state)):
            raise RuntimeError("libretro restore-state failed")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.loaded:
            self.dll.retro_unload_game()
        self.dll.retro_deinit()
        self.__class__._active_pid = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
