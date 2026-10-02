// Windows versions of the process, file and desktop helpers in util.h (the POSIX ones are in util.cpp).
#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "util.h"

namespace util {

std::wstring widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

static std::wstring win_path(const std::string& p) {
    std::wstring w = widen(p);
    for (auto& c : w)
        if (c == L'/') c = L'\\';
    return w;
}

// SearchPath looks in the app's own folder first, then the current one, the system folders and PATH, which is
// where the bundled exiv2.exe and the system's curl.exe are found.
static std::wstring find_exe(const std::string& exe) {
    std::wstring name = widen(exe);
    if (name.find(L'\\') != std::wstring::npos || name.find(L'/') != std::wstring::npos) return win_path(exe);
    wchar_t buf[MAX_PATH * 2];
    DWORD n = SearchPathW(nullptr, name.c_str(), L".exe", DWORD(std::size(buf)), buf, nullptr);
    return n > 0 && n < std::size(buf) ? std::wstring(buf, n) : L"";
}

bool which(const std::string& exe) { return !find_exe(exe).empty(); }

// One argument, quoted the way CommandLineToArgvW / the MSVC runtime parse it.
static std::wstring quote_arg(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) return a;
    std::wstring q = L"\"";
    for (size_t i = 0;; i++) {
        size_t bs = 0;
        while (i < a.size() && a[i] == L'\\') { i++; bs++; }
        if (i == a.size()) { q.append(bs * 2, L'\\'); break; }
        if (a[i] == L'"') { q.append(bs * 2 + 1, L'\\'); q += L'"'; }
        else { q.append(bs, L'\\'); q += a[i]; }
    }
    return q + L"\"";
}

ProcResult run(const std::vector<std::string>& argv, const std::string& stdin_data, int timeout_s) {
    ProcResult r;
    if (argv.empty()) return r;
    std::wstring exe = find_exe(argv[0]);
    if (exe.empty()) {
        r.rc = 127;
        r.err = "cannot run " + argv[0] + ": not found";
        return r;
    }
    std::wstring cmd = quote_arg(exe);
    for (size_t i = 1; i < argv.size(); i++) cmd += L" " + quote_arg(widen(argv[i]));
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr, err_r = nullptr, err_w = nullptr;
    if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0) || !CreatePipe(&err_r, &err_w, &sa, 0)) {
        r.err = "pipe failed";
        return r;
    }
    // our ends are not inherited
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_r, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = err_w;
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT, nullptr,
                             nullptr, &si, &pi);
    CloseHandle(in_r);
    CloseHandle(out_w);
    CloseHandle(err_w);
    if (!ok) {
        CloseHandle(in_w); CloseHandle(out_r); CloseHandle(err_r);
        r.err = fmt("cannot run %s (error %lu)", argv[0].c_str(), GetLastError());
        return r;
    }
    // Readers and the writer on their own threads: a full pipe never blocks the other side.
    auto reader = [](HANDLE h, std::string* dst) {
        char buf[65536];
        DWORD n;
        while (ReadFile(h, buf, sizeof buf, &n, nullptr) && n > 0) dst->append(buf, n);
    };
    std::thread to(reader, out_r, &r.out), te(reader, err_r, &r.err);
    std::thread ti([in_w, &stdin_data] {
        size_t off = 0;
        DWORD n;
        while (off < stdin_data.size() && WriteFile(in_w, stdin_data.data() + off, DWORD(std::min<size_t>(stdin_data.size() - off, 1 << 20)), &n, nullptr))
            off += n;
        CloseHandle(in_w);
    });
    bool timed_out = WaitForSingleObject(pi.hProcess, DWORD(timeout_s) * 1000) == WAIT_TIMEOUT;
    if (timed_out) {
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, 5000);
    }
    ti.join();
    to.join();
    te.join();
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(out_r);
    CloseHandle(err_r);
    r.rc = timed_out ? -2 : int(code);
    // Console tools (exiv2, curl, ffmpeg) end lines with \r\n here; callers parse lines, as on Linux.
    r.out = replace_all(r.out, "\r\n", "\n");
    r.err = replace_all(r.err, "\r\n", "\n");
    if (timed_out) r.err += "\n(timed out)";
    return r;
}

bool copy_file_preserve(const std::string& from, const std::string& to, std::string& err) {
    // CopyFile keeps the times and attributes; COPY_FILE_FAIL_IF_EXISTS: never overwrites anything.
    std::wstring part = win_path(to + ".part");
    if (!CopyFileExW(win_path(from).c_str(), part.c_str(), nullptr, nullptr, nullptr, COPY_FILE_FAIL_IF_EXISTS)) {
        err = fmt("copy %s: error %lu", from.c_str(), GetLastError());
        return false;
    }
    HANDLE h = CreateFileW(part.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(h);
        CloseHandle(h);
    }
    if (!MoveFileExW(part.c_str(), win_path(to).c_str(), MOVEFILE_WRITE_THROUGH)) {  // no REPLACE_EXISTING
        err = fmt("rename to %s: error %lu", to.c_str(), GetLastError());
        DeleteFileW(part.c_str());
        return false;
    }
    return true;
}

void attach_console() {
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    FILE* f;
    if (_fileno(stdout) < 0 || GetStdHandle(STD_OUTPUT_HANDLE) == nullptr) freopen_s(&f, "CONOUT$", "w", stdout);
    if (_fileno(stderr) < 0 || GetStdHandle(STD_ERROR_HANDLE) == nullptr) freopen_s(&f, "CONOUT$", "w", stderr);
    SetConsoleOutputCP(CP_UTF8);
}

bool system_reduce_motion() {
    BOOL on = TRUE;
    SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, &on, 0);
    return !on;
}

bool system_dark_mode() {  // "Choose your app mode"
    DWORD light = 1, size = sizeof light;
    RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr,
                 &light, &size);
    return light == 0;
}

bool have_trash() { return true; }  // the Recycle Bin

bool trash(const std::string& path) {
    std::wstring w = win_path(path);
    w.push_back(L'\0');  // double-NUL terminated list
    SHFILEOPSTRUCTW op{};
    op.wFunc = FO_DELETE;
    op.pFrom = w.c_str();
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT;
    return SHFileOperationW(&op) == 0 && !op.fAnyOperationsAborted;
}

void open_path(const std::string& p) {
    bool url = p.find("://") != std::string::npos;
    std::wstring w = url ? widen(p) : win_path(p);
    std::thread([w] {
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        ShellExecuteW(nullptr, L"open", w.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        CoUninitialize();
    }).detach();
}

std::string choose_folder(const std::string& title) {
    std::string out;
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) return out;
    IFileOpenDialog* dlg = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog, (void**)&dlg))) {
        DWORD opts = 0;
        dlg->GetOptions(&opts);
        dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        dlg->SetTitle(widen(title).c_str());
        if (SUCCEEDED(dlg->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item))) {
                PWSTR path = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                    out = norm_path(narrow(path));
                    CoTaskMemFree(path);
                }
                item->Release();
            }
        }
        dlg->Release();
    }
    CoUninitialize();
    return out;
}

}  // namespace util

#endif  // _WIN32
