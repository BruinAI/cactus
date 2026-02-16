Vendored libcurl layout
=======================

Place prebuilt libcurl artifacts here to bundle curl for Android, iOS, and macOS builds.

Expected structure:

```
libs/curl/
  include/
    curl/
      curl.h
      ...
  android/
    arm64-v8a/
      libcurl.a
  ios/
    device/
      libcurl.a
    simulator/
      libcurl.a
  macos/
    libcurl.a
```

Notes:

- `include/` must contain curl public headers.
- Android currently supports `arm64-v8a` only.
- iOS uses:
  - `ios/device/libcurl.a` for `iphoneos`
  - `ios/simulator/libcurl.a` for `iphonesimulator`
- You can override the default location with `-DCACTUS_CURL_ROOT=/abs/path/to/curl`.
