#!/usr/bin/env python3
from __future__ import annotations

from test_aten_op_batch_000 import CUSTOM_TEMPLATES as T000
from test_aten_op_batch_001 import CUSTOM_TEMPLATES as T001
from test_aten_op_batch_002 import CUSTOM_TEMPLATES as T002
from test_aten_op_batch_003 import CUSTOM_TEMPLATES as T003
from test_aten_op_batch_004 import CUSTOM_TEMPLATES as T004
from test_aten_op_batch_005 import CUSTOM_TEMPLATES as T005
from test_aten_op_batch_006 import CUSTOM_TEMPLATES as T006
from test_aten_op_batch_007 import CUSTOM_TEMPLATES as T007
from test_aten_op_batch_008 import CUSTOM_TEMPLATES as T008
from test_aten_op_batch_009 import CUSTOM_TEMPLATES as T009


CUSTOM_TEMPLATES = {}
for templates in (T000, T001, T002, T003, T004, T005, T006, T007, T008, T009):
    CUSTOM_TEMPLATES.update(templates)
