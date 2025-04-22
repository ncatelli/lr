# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/ncatelli/lr/releases/tag/lr-derive-v0.1.0) - 2025-04-22

### Fixed

- [**breaking**] Regression in table generation due to first set generation failing. ([#43](https://github.com/ncatelli/lr/pull/43))

### Other

- Parsing with state ([#28](https://github.com/ncatelli/lr/pull/28))
- Support lifetime annotations on symbol types ([#26](https://github.com/ncatelli/lr/pull/26))
- preallocate parse ctx stacks to 128 elems ([#19](https://github.com/ncatelli/lr/pull/19))
- bump syn dep to 2+ ([#17](https://github.com/ncatelli/lr/pull/17))
- Decouple relex from tests ([#16](https://github.com/ncatelli/lr/pull/16))
- Support tokens from relex with associated fields ([#11](https://github.com/ncatelli/lr/pull/11))
- Rename tokens, symbols and productions for consistency ([#10](https://github.com/ncatelli/lr/pull/10))
- Parser generator runtime ([#6](https://github.com/ncatelli/lr/pull/6))
