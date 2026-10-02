#include "http.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mutex>
typedef SOCKET sock_t;
#define poll WSAPoll
#define sock_close closesocket
#define sock_err() (WSAGetLastError())
#define MSG_NOSIGNAL 0
#define SOCK_CLOEXEC 0
#else
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
#define sock_err() (errno)
#endif

#include <cerrno>
#include <chrono>
#include <cstring>

#include "util.h"

std::string url_join(const std::string& base, const std::string& path) {
    std::string b = base;
    while (!b.empty() && b.back() == '/') b.pop_back();
    return b + (path.empty() || path[0] == '/' ? path : "/" + path);
}

namespace {

struct Url {
    std::string scheme, host, port, path;
};

bool parse_url(const std::string& u, Url& out) {
    auto p = u.find("://");
    if (p == std::string::npos) return false;
    out.scheme = util::lower(u.substr(0, p));
    std::string rest = u.substr(p + 3);
    auto slash = rest.find('/');
    std::string hostport = rest.substr(0, slash);
    out.path = slash == std::string::npos ? "/" : rest.substr(slash);
    if (!hostport.empty() && hostport[0] == '[') {  // [::1]:8011
        auto rb = hostport.find(']');
        out.host = hostport.substr(1, rb - 1);
        out.port = rb + 1 < hostport.size() && hostport[rb + 1] == ':' ? hostport.substr(rb + 2) : "";
    } else {
        auto colon = hostport.rfind(':');
        out.host = hostport.substr(0, colon);
        out.port = colon == std::string::npos ? "" : hostport.substr(colon + 1);
    }
    if (out.port.empty()) out.port = out.scheme == "https" ? "443" : "80";
    return !out.host.empty();
}

std::string dechunk(const std::string& s) {
    std::string out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t eol = s.find("\r\n", pos);
        if (eol == std::string::npos) break;
        size_t n = strtoul(s.substr(pos, eol - pos).c_str(), nullptr, 16);
        if (n == 0) break;
        out.append(s, eol + 2, n);
        pos = eol + 2 + n + 2;
    }
    return out;
}

HttpResponse plain_http(const std::string& method, const Url& u, const std::string& body, const std::string& bearer, int timeout_s) {
    HttpResponse r;
#ifdef _WIN32
    static std::once_flag wsa;
    std::call_once(wsa, [] { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); });
#endif
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (int e = getaddrinfo(u.host.c_str(), u.port.c_str(), &hints, &res); e != 0) {
        r.error = std::string("resolve ") + u.host + ": " + gai_strerror(e);
        return r;
    }
    sock_t fd = INVALID_SOCKET;
    int cerr = 0;
    for (auto* a = res; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd == INVALID_SOCKET) continue;
#ifdef _WIN32
        DWORD tv = DWORD(timeout_s) * 1000;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);
#else
        struct timeval tv{timeout_s, 0};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
        if (connect(fd, a->ai_addr, (int)a->ai_addrlen) == 0) break;
        cerr = sock_err();
        sock_close(fd);
        fd = INVALID_SOCKET;
    }
    freeaddrinfo(res);
    if (fd == INVALID_SOCKET) {
#ifdef _WIN32
        r.error = "connect " + u.host + ":" + u.port + ": " + (cerr == WSAECONNREFUSED ? "Connection refused" : "error " + std::to_string(cerr));
#else
        r.error = "connect " + u.host + ":" + u.port + ": " + strerror(cerr);
#endif
        return r;
    }
    std::string req = method + " " + u.path + " HTTP/1.1\r\nHost: " + u.host + ":" + u.port +
                      "\r\nConnection: close\r\nAccept: application/json\r\nUser-Agent: jev-photos\r\n";
    if (!bearer.empty()) req += "Authorization: Bearer " + bearer + "\r\n";
    if (method != "GET") req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    for (size_t off = 0; off < req.size();) {
        auto w = send(fd, req.data() + off, (int)(req.size() - off), MSG_NOSIGNAL);
        if (w <= 0) { r.error = "send failed (" + std::to_string(sock_err()) + ")"; sock_close(fd); return r; }
        off += size_t(w);
    }
    std::string raw;
    char buf[65536];
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    for (;;) {
        int left = int(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
        if (left <= 0) { r.error = "timed out"; sock_close(fd); return r; }
        struct pollfd p{fd, POLLIN, 0};
        int pr = poll(&p, 1, left);
        if (pr < 0 && errno == EINTR) continue;
        if (pr <= 0) continue;
        auto n = recv(fd, buf, (int)sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        raw.append(buf, size_t(n));
    }
    sock_close(fd);
    size_t hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos || raw.compare(0, 5, "HTTP/") != 0) { r.error = "bad HTTP response"; return r; }
    std::string head = util::lower(raw.substr(0, hdr_end));
    r.status = atoi(raw.c_str() + raw.find(' ') + 1);
    r.body = raw.substr(hdr_end + 4);
    if (head.find("transfer-encoding: chunked") != std::string::npos) r.body = dechunk(r.body);
    if (!r.ok()) r.error = "HTTP " + std::to_string(r.status);
    return r;
}

HttpResponse curl_http(const std::string& method, const std::string& url, const std::string& body, const std::string& bearer, int timeout_s) {
    HttpResponse r;
    if (!util::which("curl")) { r.error = "https needs the curl binary"; return r; }
    std::string cfg_path = util::temp_path("curl.cfg"), body_path = util::temp_path("body.json");
    auto q = [](const std::string& s) { return "\"" + util::replace_all(util::replace_all(s, "\\", "\\\\"), "\"", "\\\"") + "\""; };
    std::string cfg = "url = " + q(url) + "\nrequest = " + q(method) + "\nsilent\nshow-error\nwrite-out = \"\\n%{http_code}\"\n" +
                      "max-time = " + std::to_string(timeout_s) + "\nheader = \"Accept: application/json\"\n";
    if (!bearer.empty()) cfg += "header = " + q("Authorization: Bearer " + bearer) + "\n";
    if (method != "GET") {
        util::write_file(body_path, body, 0600);
        cfg += "header = \"Content-Type: application/json\"\ndata-binary = " + q("@" + body_path) + "\n";
    }
    util::write_file(cfg_path, cfg, 0600);
    util::ProcResult p = util::run({"curl", "-K", cfg_path}, "", timeout_s + 5);
    remove(cfg_path.c_str());
    remove(body_path.c_str());
    size_t nl = p.out.rfind('\n');
    if (p.rc != 0 || nl == std::string::npos) { r.error = "curl: " + util::trim(p.err); return r; }
    r.status = atoi(p.out.c_str() + nl + 1);
    r.body = p.out.substr(0, nl);
    if (!r.ok()) r.error = "HTTP " + std::to_string(r.status);
    return r;
}

}  // namespace

HttpResponse http_request(const std::string& method, const std::string& url, const std::string& body, const std::string& bearer, int timeout_s) {
    Url u;
    if (!parse_url(url, u)) {
        HttpResponse r;
        r.error = "bad URL: " + url;
        return r;
    }
    if (u.scheme == "https") return curl_http(method, url, body, bearer, timeout_s);
    return plain_http(method, u, body, bearer, timeout_s);
}
