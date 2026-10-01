// Minimal HTTP client: plain sockets for http://, the curl binary for https:// (keys go through a 0600 config file).
#pragma once

#include <string>

struct HttpResponse {
    int status = 0;  // 0 = transport error (see error)
    std::string body, error;
    bool ok() const { return status >= 200 && status < 300; }
};

HttpResponse http_request(const std::string& method, const std::string& url, const std::string& body,
                          const std::string& bearer, int timeout_s);
inline HttpResponse http_post_json(const std::string& url, const std::string& body, const std::string& bearer, int timeout_s) {
    return http_request("POST", url, body, bearer, timeout_s);
}
inline HttpResponse http_get(const std::string& url, const std::string& bearer, int timeout_s) {
    return http_request("GET", url, "", bearer, timeout_s);
}
std::string url_join(const std::string& base, const std::string& path);
