# Code Review Audit - ComfyUI-Koshi-Nodes

**Date:** 2026-03-11
**Branch:** `claude/code-review-audit-NDY1H`

---

## Summary

Comprehensive review of repository structure, naming consistency, code quality, and README alignment. The repo is **overall well-structured** with 18 working nodes correctly registered across 5 categories. Several issues were found and fixed; others are documented as recommendations.

---

## Issues Found & Fixed

### 1. Unused import removed
- **File:** `nodes/effects/raymarcher.py:11`
- **Issue:** `import os` was imported but never used
- **Fix:** Removed unused import

### 2. Bare `except:` anti-patterns replaced
- **Files:** `nodes/effects/raymarcher.py:131`, `nodes/image/dither/filter.py:279`
- **Issue:** Bare `except:` catches all exceptions including `SystemExit` and `KeyboardInterrupt`
- **Fix:** Changed to `except Exception:` which is safer

### 3. Display name inconsistency fixed
- **File:** `nodes/effects/glitch.py:312`
- **Issue:** Internal mapping used `"Glitch Shader Effect"` instead of convention `"░▀░ KN Glitch"`
- **Fix:** Updated to follow the prefix naming convention

### 4. README display name mismatch fixed
- **File:** `README.md:111`
- **Issue:** README listed `░▀░ KN Dither GPU` but actual node display name is `░▀░ KN Dithering Filter (GPU)`
- **Fix:** Updated README to match actual node name

### 5. Misleading "Legacy" comment corrected
- **File:** `__init__.py:22`
- **Issue:** Comment said `# Legacy - to be removed after migration` on active SIDKIT/Export node categories (`nodes.image.dither`, `nodes.image.greyscale`, `nodes.image.binary`, `nodes.export`) — these are NOT legacy
- **Fix:** Removed misleading comment, moved empty placeholder categories (`nodes.sidkit`, `nodes.audio`) to bottom with accurate comment

### 6. Function-level import moved to module level
- **File:** `nodes/generators/patterns_2d.py:267`
- **Issue:** `import inspect` was inside a function body, causing re-import on every call
- **Fix:** Moved to module-level import

### 7. f-string logging replaced with lazy formatting
- **Files:** `nodes/effects/chromatic_aberration.py:67,94`, `nodes/flux_motion/__init__.py:16,24,32`
- **Issue:** Used `f"..."` in `logger.debug()` calls — evaluates string even when debug logging is disabled
- **Fix:** Changed to `logger.debug("...: %s", e)` lazy format

---

## Node Registration Audit

### Node Count: 18 (matches README)

| Category | Nodes | Prefix | Status |
|----------|-------|--------|--------|
| Effects | Koshi Effects, Bloom, Chromatic, Glitch | `░▀░` | OK |
| Motion | Schedule, Motion Engine, Feedback | `▄▀▄` | OK |
| Generators | Glitch Candies, Shape Morph, Noise Displace, Raymarcher | `▄█▄` | OK |
| SIDKIT/Image | Binary, Greyscale, Dither, Dithering Filter (GPU) | `░▒░`/`░▀░` | OK |
| Export | OLED Screen, Sprite Sheet | `░▒░` | OK |
| Utility | Metadata | `◊` | OK |

### Empty Categories
- `nodes.sidkit` — empty stub (backwards compatibility placeholder)
- `nodes.audio` — empty stub (reserved for future)

---

## Recommendations (Not Fixed - Future Work)

### HIGH Priority - Code Duplication
- `_bayer_matrix()` is duplicated in 4 files: `koshi_effects.py`, `dither/nodes.py`, `binary/nodes.py`, `greyscale/nodes.py`
- `smoothstep()` is duplicated in 3 files: `generators/utils.py`, `flux_motion/core/interpolation.py`, `effects/glitch.py`
- **Recommendation:** Extract shared functions to `nodes/utils/image_ops.py` and `nodes/utils/math_ops.py`

### MEDIUM Priority - Inconsistent Node Key Naming
- Effects __init__.py uses `Koshi_Bloom` but bloom.py internally uses `KoshiBloomShader`
- Effects __init__.py uses `Koshi_Glitch` but glitch.py internally uses `KoshiGlitchShader`
- Only the __init__.py keys matter for ComfyUI, but internal keys should match to avoid confusion
- **Recommendation:** Standardize internal keys to match __init__.py keys

### MEDIUM Priority - Hardcoded Paths
- `nodes/utility/koshi_metadata.py:151` — hardcodes `~/ComfyUI/output/metadata` path
- Multiple files use fragile `.parent.parent.parent / "shaders"` path chains
- **Recommendation:** Use ComfyUI's folder_paths API or a shared config for paths

### LOW Priority - Performance
- Floyd-Steinberg dithering in `dither/nodes.py` and `binary/nodes.py` uses nested Python loops (slow for large images)
- Raymarching in `patterns_3d.py` has fixed 64 iterations with no early termination
- **Recommendation:** Consider NumPy vectorization for hot paths

### LOW Priority - Type Annotations
- Most node methods lack return type hints
- Some parameters like `vae` in feedback.py have no type hints
- **Recommendation:** Add type hints incrementally

---

## Project Structure Verification

README project structure matches actual layout. All directories and files documented in README exist. Workflow files match README listing (7 workflows present).

### Files present:
- `shaders/` — 5 GLSL shaders (bloom, bloom_composite, chromatic_aberration, dithering_raymarcher, image_dithering_filter)
- `js/` — 4 JS files (appearance, glitch_candies, preview, livePreview)
- `workflows/` — 7 workflow JSONs
- `tests/` — 8 test files covering imports, bug fixes, and node-specific tests

---

## Conclusion

The repository is **intact and well-organized** following the consolidation from 40 to 18 nodes. The main issues were cosmetic (naming inconsistencies, unused imports, anti-patterns) rather than structural. The node registration chain works correctly and all 18 nodes are properly registered with consistent category prefixes.
