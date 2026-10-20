# RoboSTAR source

Source: https://github.com/zyjOrz/RoboSTAR
Commit: `2371ba3e7cdd57242b4a96709a977eabc298e940`
License: Apache-2.0, retained in LICENSE.

Only the `robostar` Python package is vendored. Model weights are downloaded separately
from `Ivystream/RoboSTAR` revision `bf0ed2f3bf43b8a1d9def85ca8ab51bef5a10b06`.
The model card declares `license: other`; source-code licensing does not establish
redistribution rights for weights or training data. Keep this a local personal research demo.

Local compatibility and memory changes are documented in the root README. No fallback
dictionary renderer is used when model inference fails.
