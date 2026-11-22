"""Store the Hugging Face token in a Windows user-protected local file."""

from ctypes import (
    HRESULT, POINTER, Structure, WinDLL, WinError, byref, c_ubyte, c_void_p, cast,
    create_string_buffer, string_at, wstring_at,
)
from ctypes.wintypes import DWORD
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID


FOLDERID_LOCAL_APP_DATA = UUID("f1b32785-6fba-4fcf-9d55-7b8e7f157091")
RPC_E_CHANGED_MODE = -2147417850


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


def _local_app_data():
    ole = WinDLL("ole32")
    ole.CoInitializeEx.argtypes = [c_void_p, DWORD]
    ole.CoInitializeEx.restype = HRESULT
    ole.CoUninitialize.argtypes = []
    ole.CoUninitialize.restype = None
    ole.CoTaskMemFree.argtypes = [c_void_p]
    ole.CoTaskMemFree.restype = None

    # Balance successful initialization, including S_FALSE.
    status = ole.CoInitializeEx(None, 2)
    if status < 0 and status != RPC_E_CHANGED_MODE:
        raise WinError(status)
    address = c_void_p()
    try:
        folder_id = (c_ubyte * 16).from_buffer_copy(FOLDERID_LOCAL_APP_DATA.bytes_le)
        get_path = WinDLL("shell32").SHGetKnownFolderPath
        get_path.argtypes = [POINTER(c_ubyte), DWORD, c_void_p, POINTER(c_void_p)]
        get_path.restype = HRESULT
        result = get_path(folder_id, 0, None, byref(address))
        if result < 0:
            raise WinError(result)
        return Path(wstring_at(address))
    finally:
        ole.CoTaskMemFree(address)
        if status >= 0:
            ole.CoUninitialize()


def _token_path():
    return _local_app_data() / "Transcritor" / "huggingface.dpapi"


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
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix="hf-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encrypted)
        replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
