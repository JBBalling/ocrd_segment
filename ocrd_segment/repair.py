from __future__ import absolute_import

import os.path
from skimage import draw
from scipy.ndimage import filters, morphology
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_for_segment,
    coordinates_of_segment,
    polygon_from_points,
    points_from_polygon,
    xywh_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    PageType,
    TextRegionType,
    to_xml
)
from ocrd_models.ocrd_page_generateds import (
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
    ReadingOrderType
)
from ocrd_validators.page_validator import (
    CoordinateConsistencyError,
    CoordinateValidityError,
    PageValidator
)
from .config import OCRD_TOOL
from .project import join_polygons

TOOL = 'ocrd-segment-repair'

class RepairSegmentation(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(RepairSegmentation, self).__init__(*args, **kwargs)

    def process(self):
        """Perform generic post-processing of page segmentation with Shapely and OpenCV.
        
        Open and deserialize PAGE input files and their respective images,
        then validate syntax and semantics, checking for invalid or inconsistent
        segmentation. Fix invalidities by simplifying and/or re-ordering polygon paths.
        Fix inconsistencies by shrinking segment polygons to their parents. Log
        errors that cannot be repaired automatically.
        
        Next, if ``simplify`` is non-zero, then for each segment (top-level page or
        recursive region, line, word, glyph), simplify the polygon points up to that
        precision, while preserving its topology and parent-child consistency.
        (This will usually reduce the number of points.)
        
        \b
        Next, if ``plausibilize``, then for each segment (top-level page or recursive region)
        which contains any text regions, try to find all pairs of such regions in it that
        are redundant judging by their coordinates:
        - If near identical coordinates,
          then either region can be deleted.
        - If proper containment of one in the other,
          then the one region can be deleted.
        - If high-coverage containment of one in the other
          (i.e. a fraction of more than ``plausibilize_merge_min_overlap``),
          then the one region can be merged into the other.
        - If another overlap, find all pairs of lines from each side
          that are redundant judging by their coordinates:
           * If near identical coordinates,
             then either line can be deleted.
           * If proper containment of one over the other,
             then the one line can be deleted.
           * If high-coverage containment of one in the other
             (a fraction of more than ``plausibilize_merge_min_overlap``),
             then the one line can be merged into the other.
           * If another overlap, and
             - if either line's centroid is in the other, 
               then the smaller line can be merged into the larger,
             - otherwise the smaller line can be subtracted from the larger.
        Apply those repairs and update the reading order.
        
        Furthermore, if ``sanitize``, then for each text region, update
        the coordinates to become the minimal convex hull of its constituent
        text lines. (But consider running ocrd-segment-project instead.)
        
        Finally, produce new output files by serialising the resulting hierarchy.
        """
        LOG = getLogger('processor.RepairSegmentation')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        simplify = self.parameter['simplify']
        sanitize = self.parameter['sanitize']
        plausibilize = self.parameter['plausibilize']
        
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            
            #
            # validate segmentation (warn of children extending beyond their parents)
            #
            report = PageValidator.validate(ocrd_page=pcgts, 
                                            page_textequiv_consistency='off',
                                            check_baseline=False)
            if not report.is_valid:
                errors = report.errors
                report.errors = []
                for error in errors:
                    if isinstance(error, (CoordinateConsistencyError,CoordinateValidityError)):
                        if error.tag == 'Page':
                            element = page.get_Border()
                        elif error.tag.endswith('Region'):
                            element = next((region
                                            for region in page.get_AllRegions()
                                            if region.id == error.ID), None)
                        elif error.tag == 'TextLine':
                            element = next((line
                                            for region in page.get_AllRegions(classes=['Text'])
                                            for line in region.get_TextLine()
                                            if line.id == error.ID), None)
                        elif error.tag == 'Word':
                            element = next((word
                                            for region in page.get_AllRegions(classes=['Text'])
                                            for line in region.get_TextLine()
                                            for word in line.get_Word()
                                            if word.id == error.ID), None)
                        elif error.tag == 'Glyph':
                            element = next((glyph
                                            for region in page.get_AllRegions(classes=['Text'])
                                            for line in region.get_TextLine()
                                            for word in line.get_Word()
                                            for glyph in word.get_Glyph()
                                            if glyph.id == error.ID), None)
                        else:
                            LOG.error("Unrepairable error for unknown segment type: %s",
                                      str(error))
                            report.add_error(error)
                            continue
                        if not element:
                            LOG.error("Unrepairable error for unknown segment element: %s",
                                      str(error))
                            report.add_error(error)
                            continue
                        if isinstance(error, CoordinateConsistencyError):
                            try:
                                ensure_consistent(element)
                            except Exception as e:
                                LOG.error(str(e))
                                report.add_error(error)
                                continue
                        else:
                            ensure_valid(element)
                        LOG.warning("Fixed %s for %s '%s'", error.__class__.__name__,
                                    error.tag, error.ID)
            # show remaining errors
            if not report.is_valid:
                LOG.warning(report.to_xml())
            # simplify
            if simplify:
                self.simplify_page(page, page_id)
            # delete/merge/split redundant text regions (or its text lines)
            if plausibilize:
                self.plausibilize_page(page, page_id)
            # shrink/expand text regions to the hull of their text lines
            if sanitize:
                page_image, page_coords, _ = self.workspace.image_from_page(
                    page, page_id,
                    feature_selector='binarized',
                    feature_filter='clipped')
                shrink_regions(page_image, page_coords, page, page_id,
                               padding=self.parameter['sanitize_padding'])
            
            file_id = make_file_id(input_file, self.output_file_grp)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp,
                                            file_id + '.xml'),
                content=to_xml(pcgts))

    def simplify_page(self, page, page_id):
        if page.get_Border() is not None:
            simplify(page.get_Border(), self.parameter['simplify'])
        for region in page.get_AllRegions():
            simplify(region, self.parameter['simplify'])
            if isinstance(region, TextRegionType):
                for line in region.get_TextLine():
                    simplify(line, self.parameter['simplify'])
                    for word in line.get_Word():
                        simplify(word, self.parameter['simplify'])
                        for glyph in word.get_Glyph():
                            simplify(glyph, self.parameter['simplify'])
                            ensure_consistent(glyph, at_parent=True)
                        ensure_consistent(word, at_parent=True)
                    ensure_consistent(line, at_parent=True)
            ensure_consistent(region, at_parent=True)
        if page.get_Border() is not None:
            ensure_consistent(page)
    
    def plausibilize_page(self, page, page_id):
        ro = page.get_ReadingOrder()
        if ro:
            rogroup = ro.get_OrderedGroup() or ro.get_UnorderedGroup()
        else:
            rogroup = None
        marked_for_deletion = list() # which regions/lines will get removed?
        marked_for_merging = dict() # which regions/lines will get merged into which regions/lines?
        marked_for_splitting = dict() # which regions/lines will get split along which regions/lines?
        # cover recursive region structure (but compare only at the same level)
        parents = {region.parent_object_
                   for region in page.get_AllRegions(classes=['Text'])}
        for parent in parents:
            regions = parent.get_TextRegion()
            # sort by area to ensure to arrive at a total ordering compatible
            # with the topological sort along containment/equivalence arcs
            # (so we can avoid substituting regions with superregions that have
            #  themselves been substituted/deleted):
            regpolys = sorted([(region, Polygon(polygon_from_points(region.get_Coords().points)))
                               for region in regions],
                              key=lambda x: x[1].area)
            for i in range(0, len(regpolys)):
                for j in range(i+1, len(regpolys)):
                    region1 = regpolys[i][0]
                    region2 = regpolys[j][0]
                    regpoly1 = regpolys[i][1]
                    regpoly2 = regpolys[j][1]
                    if _compare_segments(region1, region2, regpoly1, regpoly2,
                                         marked_for_deletion, marked_for_merging,
                                         self.parameter['plausibilize_merge_min_overlap'],
                                         page_id):
                        # non-trivial overlap: mutually plausibilize lines
                        linepolys1 = sorted([(line, Polygon(polygon_from_points(line.get_Coords().points)))
                                             for line in region1.get_TextLine()],
                                            key=lambda x: x[1].area)
                        linepolys2 = sorted([(line, Polygon(polygon_from_points(line.get_Coords().points)))
                                             for line in region2.get_TextLine()],
                                            key=lambda x: x[1].area)
                        for line1, linepoly1 in linepolys1:
                            for line2, linepoly2 in linepolys2:
                                if _compare_segments(line1, line2, linepoly1, linepoly2,
                                                     marked_for_deletion, marked_for_merging,
                                                     self.parameter['plausibilize_merge_min_overlap'],
                                                     page_id):
                                    # non-trivial overlap: check how close to each other
                                    if (linepoly1.centroid.within(linepoly2) or
                                        linepoly2.centroid.within(linepoly1)):
                                        # merge lines and regions
                                        if regpoly1.area > regpoly2.area:
                                            marked_for_merging[line2.id] = line1
                                            marked_for_merging[region2.id] = region1
                                        else:
                                            marked_for_merging[line1.id] = line2
                                            marked_for_merging[region1.id] = region2
                                    else:
                                        # split in favour of line with larger share
                                        if linepoly1.area < linepoly2.area:
                                            marked_for_splitting[line2.id] = line1
                                        else:
                                            marked_for_splitting[line1.id] = line2
                    elif marked_for_merging.get(region1.id, None) == region2:
                        marked_for_deletion.extend([line.id for line in region1.get_TextLine()])
                    elif marked_for_merging.get(region2.id, None) == region1:
                        marked_for_deletion.extend([line.id for line in region2.get_TextLine()])
            # apply everything, passing the regions sorted (see above)
            _plausibilize_segments(regpolys, rogroup,
                                   marked_for_deletion,
                                   marked_for_merging,
                                   marked_for_splitting)
    
def _compare_segments(seg1, seg2, poly1, poly2, marked_for_deletion, marked_for_merging, min_overlap, page_id):
    """Determine redundancies in a pair of regions/lines
    
    \b
    For segments ``seg1`` (with coordinates ``poly1``) and ``seg2`` (with coordinates ``poly2``),
    - if their coordinates are nearly identical, then just mark ``seg2`` for deletion
    - if either properly contains the other, then mark the other for deletion
    - if they overlap, then mark the most overlapped side in favour of the other – unless
      - the union is larger than the sum (i.e. covers area outside of both) and
      - the intersection is smaller than ``min_overlap`` fraction of either side
    
    Return whether something else besides deletion must be done about the redundancy,
    i.e. true iff they overlap, but neither side could be marked for deletion.
    """
    LOG = getLogger('processor.RepairSegmentation')
    # LOG.debug('Comparing %s and %s',
    #           '%s "%s"' % (_tag_name(seg1), seg1.id),
    #           '%s "%s"' % (_tag_name(seg2), seg2.id))
    if poly1.almost_equals(poly2):
        LOG.debug('Page "%s" %s is almost equal to %s', page_id,
                  '%s "%s"' % (_tag_name(seg2), seg2.id),
                  '%s "%s"' % (_tag_name(seg1), seg1.id))
        marked_for_deletion.append(seg2.id)
    elif poly1.contains(poly2):
        LOG.debug('Page "%s" %s is within %s', page_id,
                  '%s "%s"' % (_tag_name(seg2), seg2.id),
                  '%s "%s"' % (_tag_name(seg1), seg1.id))
        marked_for_deletion.append(seg2.id)
    elif poly2.contains(poly1):
        LOG.debug('Page "%s" %s is within %s', page_id,
                  '%s "%s"' % (_tag_name(seg1), seg1.id),
                  '%s "%s"' % (_tag_name(seg2), seg2.id))
        marked_for_deletion.append(seg1.id)
    elif poly1.overlaps(poly2):
        inter_poly = poly1.intersection(poly2)
        union_poly = poly1.union(poly2)
        LOG.debug('Page "%s" %s overlaps %s by %.2f/%.2f', page_id,
                  '%s "%s"' % (_tag_name(seg1), seg1.id),
                  '%s "%s"' % (_tag_name(seg2), seg2.id),
                  inter_poly.area/poly1.area, inter_poly.area/poly2.area)
        if union_poly.convex_hull.area >= poly1.area + poly2.area:
            # skip this pair -- combined polygon encloses previously free segments
            return True
        elif inter_poly.area / poly2.area > min_overlap:
            LOG.debug('Page "%s" %s belongs to %s', page_id,
                      '%s "%s"' % (_tag_name(seg2), seg2.id),
                      '%s "%s"' % (_tag_name(seg1), seg1.id))
            marked_for_merging[seg2.id] = seg1
        elif inter_poly.area / poly1.area > min_overlap:
            LOG.debug('Page "%s" %s belongs to %s', page_id,
                      '%s "%s"' % (_tag_name(seg1), seg1.id),
                      '%s "%s"' % (_tag_name(seg2), seg2.id))
            marked_for_merging[seg1.id] = seg2
        else:
            return True

    return False

def _merge_segments(seg, superseg, poly, superpoly, segpolys, reading_order):
    """Merge one segment into another and update reading order refs.
    
    \b
    Given a region/line ``seg`` that should be dissolved into a
    region/line ``superseg``, update the latter's
    - coordinates by building a union of both and ensuring validity, and
    - sub-elements (i.e. lines of a region, words of a line) by
      determining the relative order of both in either ``reading_order``
      or in the list of ``segpolys``, and then concatenating both
      lists in that order.

    Beyond that, warn if any information is lost that would not be
    possible to merge: attributes like orientation, type, script, language
    and elements like TextStyle and TextEquiv, if different between ``seg``
    and ``superseg``.
    """
    LOG = getLogger('processor.RepairSegmentation')
    LOG.info('Merging %s "%s" into %s "%s"', 
             _tag_name(seg), seg.id, 
             _tag_name(superseg), superseg.id)
    # granularity will necessarily be lost here --
    # this is not for workflows/processors that already
    # provide good/correct segmentation and reading order
    # (in which case orientation, script and style detection
    #  can be expected as well), but rather as a postprocessor
    # for suboptimal segmentation (possibly before reading order
    # detection/correction); hence, all we now do here is
    # show warnings when granularity is lost; but there might
    # be good reasons to do more here when we have better processors
    # and use-cases in the future
    superpoly = merge_poly(superpoly, poly)
    superpoly = superpoly.exterior.coords[:-1] # keep open
    superseg.get_Coords().set_points(points_from_polygon(superpoly))
    # FIXME should we merge/mix attributes and features?
    if hasattr(seg, 'TextLine') and seg.get_TextLine():
        LOG.info('Merging region "{}" with {} text lines into "{}" with {}'.format(
            seg.id, len(seg.get_TextLine()),
            superseg.id, len(superseg.get_TextLine())))
        if (seg.id in reading_order and
            superseg.id in reading_order and
            hasattr(reading_order[seg.id], 'index') and
            hasattr(reading_order[superseg.id], 'index')):
            order = reading_order[seg.id].get_index() < reading_order[superseg.id].get_index()
        else:
            pos = next(i for i, segpoly in enumerate(segpolys) if segpoly[0] == seg)
            superpos = next(i for i, segpoly in enumerate(segpolys) if segpoly[0] == superseg)
            order = pos < superpos
        if order:
            superseg.TextLine = seg.TextLine + superseg.TextLine
        else:
            superseg.TextLine = superseg.TextLine + seg.TextLine
    elif hasattr(seg, 'Word') and seg.get_Word():
        LOG.info('Merging line "{}" with {} words into "{}" with {}'.format(
            seg.id, len(seg.get_Word()),
            superseg.id, len(superseg.get_Word())))
        pos = next(i for i, segpoly in enumerate(segpolys) if segpoly[0] == seg)
        superpos = next(i for i, segpoly in enumerate(segpolys) if segpoly[0] == superseg)
        order = pos < superpos
        if order:
            superseg.Word = seg.Word + superseg.Word
        else:
            superseg.Word = superseg.Word + seg.Word
    if hasattr(seg, 'orientation') and seg.get_orientation() != superseg.get_orientation():
        LOG.warning('Merging "{}" with orientation {} into "{}" with {}'.format(
            seg.id, seg.get_orientation(),
            superseg.id, superseg.get_orientation()))
    if hasattr(seg, 'type_') and seg.get_type() != superseg.get_type():
        LOG.warning('Merging "{}" with type {} into "{}" with {}'.format(
            seg.id, seg.get_type(),
            superseg.id, superseg.get_type()))
    if seg.get_primaryScript() != superseg.get_primaryScript():
        LOG.warning('Merging "{}" with primaryScript {} into "{}" with {}'.format(
            seg.id, seg.get_primaryScript(),
            superseg.id, superseg.get_primaryScript()))
    if seg.get_primaryLanguage() != superseg.get_primaryLanguage():
        LOG.warning('Merging "{}" with primaryLanguage {} into "{}" with {}'.format(
            seg.id, seg.get_primaryLanguage(),
            superseg.id, superseg.get_primaryLanguage()))
    if seg.get_TextStyle():
        LOG.warning('Merging "{}" with TextStyle {} into "{}" with {}'.format(
            seg.id, seg.get_TextStyle(), # FIXME needs repr...
            superseg.id, superseg.get_TextStyle())) # ...to be informative
    if seg.get_TextEquiv():
        LOG.warning('Merging "{}" with TextEquiv {} into "{}" with {}'.format(
            seg.id, seg.get_TextEquiv(), # FIXME needs repr...
            superseg.id, superseg.get_TextEquiv())) # ...to be informative
        
def _plausibilize_segments(segpolys, rogroup, marked_for_deletion, marked_for_merging, marked_for_splitting):
    """Remove redundancy among a set of segments by applying deletion/merging/splitting
    
    \b
    Given the segment-polygon tuples ``segpolys`` and analysis of actions to be taken:
    - ``marked_for_deletion``: list of segment identifiers that can be removed,
    - ``marked_for_merging``: dict of segment identifiers that can be dissolved into some other,
    - ``marked_for_splitting``: dict of segment identifiers that can be shrinked in favour of some other,
    apply these one by one (possibly recursing from regions to lines).
    
    Finally, update the reading order ``rogroup`` accordingly.
    """
    LOG = getLogger('processor.RepairSegmentation')
    wait_for_deletion = list()
    reading_order = dict()
    page_get_reading_order(reading_order, rogroup)
    for seg, poly in segpolys:
        if isinstance(seg, TextRegionType):
            # plausibilize lines first
            _plausibilize_segments([(line, Polygon(polygon_from_points(line.get_Coords().points)))
                                 for line in seg.get_TextLine()], None,
                                marked_for_deletion, marked_for_merging, marked_for_splitting)
        delete = seg.id in marked_for_deletion
        merge = seg.id in marked_for_merging
        split = seg.id in marked_for_splitting
        if split:
            otherseg = marked_for_splitting[seg.id]
            LOG.info('Shrinking %s "%s" in favour of %s "%s"', 
                     _tag_name(seg), seg.id, 
                     _tag_name(otherseg), otherseg.id)
            otherpoly = Polygon(polygon_from_points(otherseg.get_Coords().points))
            poly = poly.difference(otherpoly)
            if poly.type == 'MultiPolygon':
                poly = poly.convex_hull
            if poly.minimum_clearance < 1.0:
                poly = Polygon(np.round(poly.exterior.coords))
            poly = make_valid(poly)
            poly = poly.exterior.coords[:-1] # keep open
            seg.get_Coords().set_points(points_from_polygon(poly))
        elif delete or merge:
            if merge:
                # merge region with super region:
                superseg = marked_for_merging[seg.id]
                superpoly = Polygon(polygon_from_points(superseg.get_Coords().points))
                _merge_segments(seg, superseg, poly, superpoly, segpolys, reading_order)
            wait_for_deletion.append(seg)
            if seg.id in reading_order:
                regionref = reading_order[seg.id]
                # TODO: re-assign regionref.continuation and regionref.type to other?
                # could be any of the 6 types above:
                regionrefs = regionref.parent_object_.__getattribute__(regionref.__class__.__name__.replace('Type', ''))
                # remove in-place
                regionrefs.remove(regionref)
                if hasattr(regionref, 'index'):
                    # re-index the reading order group!
                    regionref.parent_object_.sort_AllIndexed()
    # apply actual deletions in the hierarchy
    for seg in wait_for_deletion:
        if seg.parent_object_:
            # remove in-place
            LOG.info('Deleting %s "%s"', _tag_name(seg), seg.id)
            getattr(seg.parent_object_, 'get_' + _tag_name(seg))().remove(seg)

def page_get_reading_order(ro, rogroup):
    """Add all elements from the given reading order group to the given dictionary.
    
    Given a dict ``ro`` from layout element IDs to ReadingOrder element objects,
    and an object ``rogroup`` with additional ReadingOrder element objects,
    add all references to the dict, traversing the group recursively.
    """
    regionrefs = list()
    # the reading order does not have to include all regions
    # but it may include all types of regions!
    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRefIndexed() +
                      rogroup.get_OrderedGroupIndexed() +
                      rogroup.get_UnorderedGroupIndexed())
    if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRef() +
                      rogroup.get_OrderedGroup() +
                      rogroup.get_UnorderedGroup())
    for elem in regionrefs:
        ro[elem.get_regionRef()] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            # recursive reading order element (un/ordered group):
            page_get_reading_order(ro, elem)

def shrink_regions(page_image, page_coords, page, page_id, padding=0):
    """Shrink each region outline to become the minimal concave hull of its binary foreground."""
    LOG = getLogger('processor.RepairSegmentation')
    page_array = ~ np.array(page_image.convert('1'))
    page_polygon = page_poly(page)
    if page.get_Border():
        page_polygon = Polygon(polygon_from_points(page.get_Border().get_Coords().points))
    # estimate glyph scale (roughly)
    _, components = cv2.connectedComponents(page_array.astype(np.uint8))
    _, counts = np.unique(components, return_counts=True)
    if counts.shape[0] > 1:
        counts = np.sqrt(3 * counts)
        counts = counts[(5 < counts) & (counts < 100)]
        scale = int(np.median(counts))
    else:
        scale = 43
    for region in page.get_AllRegions():
        #LOG.info('Shrinking region "%s"', region.id)
        region_mask = np.zeros_like(page_array, dtype=np.bool)
        region_polygon = coordinates_of_segment(region, page_image, page_coords)
        region_mask[draw.polygon(region_polygon[:, 1],
                                 region_polygon[:, 0],
                                 region_mask.shape)] = True
        # find outer contour (parts):
        region_array = page_array & region_mask
        contours, _ = cv2.findContours(region_array.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # determine areas of parts:
        areas = [cv2.contourArea(contour) for contour in contours]
        total_area = sum(areas)
        if not total_area:
            # ignore if too small
            LOG.warning('Zero contour area in region "%s"', region.id)
            continue
        # pick contour and convert to absolute:
        region_polygon = join_polygons([make_valid(Polygon(contour[:, 0, ::]))
                                        for contour in contours
                                        if len(contour) >= 3], loc=region.id, scale=scale)
        if padding:
            region_polygon = region_polygon.buffer(padding)
        region_polygon = coordinates_for_segment(region_polygon.exterior.coords[:-1], page_image, page_coords)
        region_polygon = Polygon(region_polygon)
        if not region_polygon.within(page_polygon):
            region_polygon = clip_poly(region_polygon, page_polygon)
        if region_polygon is not None:
            LOG.debug('Using new coordinates for region "%s"', region.id)
            region.get_Coords().set_points(points_from_polygon(region_polygon.exterior.coords[:-1]))

def simplify(segment, tolerance=0):
    if tolerance <= 0:
        return # nothing to do
    coords = segment.get_Coords()
    polygon = polygon_from_points(coords.points)
    if len(polygon) < 10:
        return # already simple enough
    poly = Polygon(polygon)
    if poly.is_empty:
        return
    if poly.length < tolerance * 4:
        return # already too small
    if poly.length > tolerance * len(polygon):
        return # already fewer points
    if np.sqrt(poly.area) < 0.1 * tolerance:
        return # already relatively small
    # simplify may easily clip out foreground
    #poly = poly.simplify(tolerance)
    # get equidistant list of points along hull
    poly = Polygon([poly.exterior.interpolate(dist).coords[0] # .xy
                    for dist in np.arange(0, poly.length, tolerance)])
    if poly.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        poly = Polygon(np.round(poly.exterior.coords))
    poly = make_valid(poly)
    polygon = poly.exterior.coords[:-1]
    coords.set_points(points_from_polygon(polygon))

def merge_poly(poly1, poly2):
    poly = poly1.union(poly2)
    if poly.type == 'MultiPolygon':
        #poly = poly.convex_hull
        poly = join_polygons(poly.geoms)
    if poly.minimum_clearance < 1.0:
        poly = Polygon(np.round(poly.exterior.coords))
    poly = make_valid(poly)
    return poly

def clip_poly(poly1, poly2):
    poly = poly1.intersection(poly2)
    if poly.is_empty or poly.area == 0.0:
        return None
    if poly.type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        poly = unary_union([geom for geom in poly.geoms if geom.area > 0])
    if poly.type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        #poly = poly.convex_hull
        poly = join_polygons(poly.geoms)
    if poly.minimum_clearance < 1.0:
        # follow-up calculations will necessarily be integer;
        # so anticipate rounding here and then ensure validity
        poly = Polygon(np.round(poly.exterior.coords))
        poly = make_valid(poly)
    return poly
    
def page_poly(page):
    return Polygon([[0, 0],
                    [0, page.get_imageHeight()],
                    [page.get_imageWidth(), page.get_imageHeight()],
                    [page.get_imageWidth(), 0]])

# same as polygon_for_parent pattern in other processors
def ensure_consistent(child, at_parent=False):
    """Make segment coordinates fit into parent coordinates.
    
    Ensure that the coordinate polygon of ``child`` is fully
    contained in the coordinate polygon of its parent.
    
    \b
    To achieve that when necessary, either
    - enlarge the parent to the union of both,
      if ``at_parent``
    - shrink the child to the intersection of both,
      otherwise.
    
    In any case, ensure the resulting polygon is valid.
    
    If the parent is at page level, and there is no Border,
    then use the page frame (and assume `at_parent=False`).
    
    If ``child`` is at page level, and there is a Border,
    then use the page frame as parent (and assume `at_parent=False`).
    """
    if isinstance(child, PageType):
        if not child.get_Border():
            return
        childp = Polygon(polygon_from_points(child.get_Border().get_Coords().points))
        parentp = page_poly(child)
        at_parent = False # clip to page frame
        parent = child
    else:
        points = child.get_Coords().points
        polygon = polygon_from_points(points)
        parent = child.parent_object_
        childp = Polygon(polygon)
        if isinstance(parent, PageType):
            if parent.get_Border():
                parentp = Polygon(polygon_from_points(parent.get_Border().get_Coords().points))
            else:
                parentp = page_poly(parent)
                at_parent = False # clip to page frame
        else:
            parentp = Polygon(polygon_from_points(parent.get_Coords().points))
    # ensure input coords have valid paths (without self-intersection)
    # (this can happen when shapes valid in floating point are rounded)
    childp = make_valid(childp)
    parentp = make_valid(parentp)
    if childp.within(parentp):
        return
    # enlargement/clipping is necessary
    if at_parent:
        # enlarge at parent
        unionp = merge_poly(childp, parentp)
        polygon = unionp.exterior.coords[:-1] # keep open
        points = points_from_polygon(polygon)
        parent.get_Coords().set_points(points)
    else:
        # clip to parent
        interp = clip_poly(childp, parentp)
        if interp is None:
            raise Exception("Segment '%s' does not intersect its parent '%s'" % (
                child.id, parent.id))
        polygon = interp.exterior.coords[:-1] # keep open
        points = points_from_polygon(polygon)
        child.get_Coords().set_points(points)

def ensure_valid(element):
    changed = False
    coords = element.get_Coords()
    points = coords.points
    polygon = polygon_from_points(points)
    array = np.array(polygon, np.int)
    if array.min() < 0:
        array = np.maximum(0, array)
        changed = True
    if array.shape[0] < 3:
        array = np.concatenate([
            array, array[::-1] + 1])
        changed = True
    polygon = array.tolist()
    poly = Polygon(polygon)
    if not poly.is_valid:
        poly = make_valid(poly)
        polygon = poly.exterior.coords[:-1]
        changed = True
    if changed:
        points = points_from_polygon(polygon)
        coords.set_points(points)

def make_valid(polygon):
    """Ensures shapely.geometry.Polygon object is valid by repeated simplification"""
    for split in range(1, len(polygon.exterior.coords)-1):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(polygon.exterior.coords[-split:]+polygon.exterior.coords[:-split])
    for tolerance in range(1, int(polygon.area)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance)
    return polygon

def _tag_name(element):
    return element.__class__.__name__[0:-4]
