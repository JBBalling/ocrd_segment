from __future__ import absolute_import

import os.path
import os
import math
import itertools
import multiprocessing as mp
from rapidfuzz import fuzz, process as fuzz_process

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    TextRegionType,
    TextLineType,
    WordType,
    TextEquivType,
    to_xml
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from maskrcnn_cli.formdata import FIELDS

from .config import OCRD_TOOL

TOOL = "ocrd-segment-mark-numbers-and-text"

class MarkNumbersAndText(Processor):
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(MarkNumbersAndText, self).__init__(*args, **kwargs)
        

    def process(self):
        LOG = getLogger('processor.MarkNumbersAndText')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        # pylint: disable=attribute-defined-outside-init
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            
            page = pcgts.get_Page()
            def mark_segment(segment, category, subtype='context'):
                custom = segment.get_custom() or ''
                if custom:
                    custom += ','
                custom += 'subtype:%s=%s' % (subtype, category)
                segment.set_custom(custom)

            allregions = page.get_AllRegions(classes=['Text'], depth=2)
            for region in allregions:
                for line in region.get_TextLine():
                    words = line.get_Word() or []
                    print(words)