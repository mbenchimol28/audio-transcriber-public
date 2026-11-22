"""Store the Hugging Face token in a Windows user-protected local file."""

from ctypes import POINTER, Structure, WinDLL, WinError, byref, c_void_p, cast, create_string_buffer, string_at
from ctypes.wintypes import DWORD
from os import environ, replace
from pathlib import Path
from tempfile import NamedTemporaryFile


class DataBlob(Structure):
    _fields_ = [("size", DWORD), ("data", c_void_p)]


def _dpapi(data, decrypt=False):
    buffer = create_string_buffer(data)
    source = DataBlob(len(data), cast(buffer, c_void_p))
    result = DataBlob()
    crypt = WinDLL("crypt32", use_last_error=True)
    operation = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    operation.argtypes = [POINTER(DataBlob), c_void_p, c_void_p, c_void_p, c_void_p, DWORD, POINTER(DataBlob)]
    if not operation(byref(source), None, None, None, None, 1, byref(result)):
        raise WinError()
    try:
        return string_at(result.data, result.size)
    finally:
        free = WinDLL("kernel32").LocalFree
        free.argtypes = [c_void_p]
        free.restype = c_void_p
        free(result.data)


def _token_path():
    return Path(environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Transcritor" / "huggingface.dpapi"


def load_token():
    try:
        encrypted = _token_path().read_bytes()
    except FileNotFoundError:
        return ""
    return _dpapi(encrypted, decrypt=True).decode("utf-8")


def save_token(token):
    path = _token_path()
    if not token:
        path.unlink(missing_ok=True)
        return
    encrypted = _dpapi(token.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix="hf-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(encrypted)
        replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
