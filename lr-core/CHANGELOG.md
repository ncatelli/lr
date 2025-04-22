# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/ncatelli/lr/releases/tag/lr-core-v0.1.0) - 2025-04-22

### Fixed

- address clippy lints in lr1 grammar
- correct clippy lints in GrammarTable display
- correct typos in docstrings
- [**breaking**] Regression in table generation due to first set generation failing. ([#43](https://github.com/ncatelli/lr/pull/43))

### Other

- Remove redundant iteration and branching in canonical collection ([#46](https://github.com/ncatelli/lr/pull/46))
- Remove repeated iteration of old closure items ([#45](https://github.com/ncatelli/lr/pull/45))
- introduce a NoOp hasher to prevent redundant hashing. ([#44](https://github.com/ncatelli/lr/pull/44))
- Fix redundant first_set generation in closure ([#42](https://github.com/ncatelli/lr/pull/42))
- Fix set generation when lookheads are out of order ([#40](https://github.com/ncatelli/lr/pull/40))
- Implement ordered sets for item set and collections ([#38](https://github.com/ncatelli/lr/pull/38))
- covert collect to to_vec() method ([#36](https://github.com/ncatelli/lr/pull/36))
- Hash on collection insertion to prevent early exit and followset fixes ([#35](https://github.com/ncatelli/lr/pull/35))
- Remove uses of context-unaware followsets from closure generation ([#33](https://github.com/ncatelli/lr/pull/33))
- Parsing with state ([#28](https://github.com/ncatelli/lr/pull/28))
- Support lifetime annotations on symbol types ([#26](https://github.com/ncatelli/lr/pull/26))
- Support tokens from relex with associated fields ([#11](https://github.com/ncatelli/lr/pull/11))
- Rename tokens, symbols and productions for consistency ([#10](https://github.com/ncatelli/lr/pull/10))
- Parser generator runtime ([#6](https://github.com/ncatelli/lr/pull/6))
